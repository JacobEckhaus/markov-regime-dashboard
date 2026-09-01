import tkinter as tk 
from tkinter import ttk, messagebox
import numpy as np
import threading
import time
import random
from datetime import datetime
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


CONTRACT_ERROR_CODES = {200, 162, 321, 354}

CONNECTION_ERROR_CODES = {326, 502, 503, 504, 1100, 1300}

class IBAPP(EWrapper, EClient): 

    def __init__(self, callback=None, error_callback=None, connection_error_callback=None):
        EClient.__init__(self, self)
        self.connected = False
        self.callback = callback
        self.error_callback = error_callback
        self.connection_error_callback = connection_error_callback
        self.last_price = None
        self.bid_price = None
        self.ask_price = None
        self.historical_data = {}
        self.hist_done = threading.Event()

        self.connected_event = threading.Event()

    def error(self, reqId, errorTime, errorCode=None, errorString=None, advancedOrderRejectJson=""):

        if errorCode is None:
   
            errorCode = errorTime
            errorTime = None

        # Filter out standard connection status notifications
        if errorCode in [2104, 2106, 2158, 2176]:
            return 
        
        if errorCode == 10167: 
            print("Note: Using Delayed Market Data")
            return
        
        print(f"Error | ReqID: {reqId} | ErrorCode: {errorCode}")
        print(f"Msg: {errorString}")


        if errorCode in CONNECTION_ERROR_CODES and self.connection_error_callback:
            self.connection_error_callback(errorCode, errorString)

        if errorCode in CONTRACT_ERROR_CODES and self.error_callback:
            self.error_callback(errorCode, errorString)
 
            self.hist_done.set()

    def nextValidId(self, orderId):
        self.connected = True
        self.connected_event.set()
        print(f"Connected to TWS. Next Valid Order ID: {orderId}")
    
    def historicalData(self, reqID, bar):
        if reqID not in self.historical_data: 
            self.historical_data[reqID] = []
        self.historical_data[reqID].append({'o':bar.open, 'h':bar.high, 'l':bar.low, 'c':bar.close})

    def historicalDataEnd(self, reqID, start, end):
        self.hist_done.set()

    def tickPrice(self, reqID, tickType, price, attrib):
        if price <= 0:
            return 
        
      
        if tickType in [4, 68]:
            self.last_price = price
            if self.callback:
                self.callback('price', price, datetime.now())
        elif tickType in [1, 66]: 
            self.bid_price = price
        elif tickType in [2, 67]:
            self.ask_price = price

    def tickSize(self, reqID, tickType, size):
        return 
    
    def tickString(self, reqID, tickType, value):
        pass

class OHLCBar:

    def __init__(self, timestamp, open_price):
        self.timestamp = timestamp
        self.open = open_price
        self.high = open_price
        self.low = open_price
        self.close = open_price
        self.tick_count = 1
        self.regime = 0 

    def update(self, price):
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.tick_count += 1

    @property
    def volatility(self):
        return (self.high - self.low) / self.close if self.close > 0 else 0 
    
class MarkovRegime:

    def __init__(self):
        self.n_states = 3
        self.current_state = 0
        self.colors = ['#3fb950', '#d29922', '#f85140']
        self.bg_colors = ['#1a3d1a', '#3d3319', '#3d1a1a']
        self.state_probs = np.array([1/3, 1/3, 1/3])
        self.transition_matrix = np.array([
            [.9, .08, .02],
            [.1, .8, .1],
            [.02, .08, .9]
        ])
        self.emission_means = np.array([.0005, .002, .005])
        self.emission_stds = np.array([.0003, .001, .003])

    def calibrate(self, hist_bars):
        if len(hist_bars) < 20: 
            return
        
        vols = np.array([(b['h'] - b['l']) / b['c'] if b['c'] > 0 else 0 for b in hist_bars])
        vols = vols[vols > 0]

        if len(vols) < 20: 
            return
        
        p33, p67 = np.percentile(vols, 33), np.percentile(vols, 67)

        regime_assignments = np.zeros(len(vols), dtype=int)
        regime_assignments[vols >= p33] = 1
        regime_assignments[vols >= p67] = 2

        for regime in range(self.n_states):
            regime_vols = vols[regime_assignments == regime]
            if len(regime_vols) >= 3:
                self.emission_means[regime] = np.mean(regime_vols)
                self.emission_stds[regime] = max(np.std(regime_vols), 1e-6)

        transition_counts = np.zeros((self.n_states, self.n_states))
        for t in range(1, len(regime_assignments)):
            prev_regime = regime_assignments[t-1]
            curr_regime = regime_assignments[t]
            transition_counts[prev_regime, curr_regime] += 1

        for i in range(self.n_states): 
            row_sum = transition_counts[i].sum()
            if row_sum > 0:
                self.transition_matrix[i] = (transition_counts[i] + .1) / (row_sum + .3)

 
        sorted_indices = np.argsort(self.emission_means)
        if not np.array_equal(sorted_indices, np.arange(self.n_states)):
            self.emission_means = self.emission_means[sorted_indices]
            self.emission_stds = self.emission_stds[sorted_indices]
            self.transition_matrix = self.transition_matrix[sorted_indices][:, sorted_indices]

        self.state_probs = np.array([1/3, 1/3, 1/3])
        print(f'Calibrated emission means: {self.emission_means}')
        print(f'Calibrated emission stds: {self.emission_stds}')

    def _gaussian_likelihood(self, vol, regime):
        mean = self.emission_means[regime]
        std = self.emission_stds[regime]
        coef = 1 / (std * np.sqrt(2 * np.pi)) 
        exponent = -.5 * ((vol - mean) /  std) ** 2
        return coef * np.exp(exponent)

    def get_regime(self, bars):
        if not bars: 
            return self.current_state
        
        current_bar = bars[-1]
        vol = current_bar.volatility

        if vol <= 0:
            current_bar.regime = self.current_state
            return self.current_state
        
        prior_probs = self.transition_matrix.T @ self.state_probs

        likelihoods = np.array([self._gaussian_likelihood(vol, i) for i in range(self.n_states)])

        posterior_probs = prior_probs * likelihoods

        prob_sum = posterior_probs.sum()
        if prob_sum > 0:
            posterior_probs = posterior_probs / prob_sum
        else: 
            print("Error normalizing the posterior")
            posterior_probs = prior_probs
        
        self.state_probs = posterior_probs
        self.current_state = int(np.argmax(posterior_probs))
        current_bar.regime = self.current_state

        return self.current_state

class LiveMarketDashboard:

    def __init__(self, root):
        self.root = root
        self.root.title('Live Market Data Dashboard')
        self.root.geometry('1200x800')
        self.root.configure(bg='#0d1117')

        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.configure_dark_theme()

        self.ib_app = IBAPP(callback=self.on_tick_data, error_callback=self.on_ib_error,
                            connection_error_callback=self.on_connection_error)
        self.connected = False
        self.streaming = False
        self.connecting = False  
 
        self.client_id = random.randint(1000, 9999)

        self.bar_duration = 5
        self.max_bars = 10
        self.ohlc_bars = deque(maxlen=self.max_bars)
        self.current_bar = None
        self.bar_start_time = None
        self.price_history = deque(maxlen=100)
        self.last_update_time = None
        self.regime_model = MarkovRegime()

        self.bar_lock = threading.Lock()
        self.update_thread = None
        self.running = False

        self.setup_ui()
        self.setup_chart()

    def configure_dark_theme(self):
        bg_color = '#0d1117'
        fg_color = '#c9d1d9'
        entry_bg = '#161b22'

        self.style.configure('TFrame', background=bg_color)
        self.style.configure('TLabelframe', background=bg_color, foreground=fg_color)
        self.style.configure('TLabelframe.Label', background=bg_color, foreground=fg_color,
                             font=('Segoe UI', 10, 'bold'))
        self.style.configure('TLabel', background=bg_color, foreground=fg_color,
                             font=('Segoe UI', 10))
        self.style.configure('Button', background=bg_color, foreground=fg_color,
                             font=('Segoe UI', 9, 'bold'), padding=(10, 5))
        self.style.map('TButton',
                       background=[('active', '#2ea043'), ('disabled', '#21262d')])
        self.style.configure('TEntry', fieldbackground=entry_bg, foreground=fg_color, 
                             insertcolor=fg_color)
        self.style.configure('Accent.TButton', background='#da3633', foreground='white')
        self.style.map('Accent.TButton',
                       background=[('active', '#f85149'), ('disabled', '#21262d')])
        
    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding='15')
        main_frame.grid(row=0, column=0, sticky='nswe')

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)

        header_frame = ttk.Frame(main_frame)
        header_frame.grid(row=0, column=0, sticky='ew', pady=(0, 15))
        
        title_label = tk.Label(header_frame, text="Live Regime Switching",
                               font=('JetBrains Mono', 18, 'bold'),
                               bg='#0d1117', fg='#58a6ff')
        title_label.pack(side='left')

        self.status_indicator = tk.Label(header_frame, text='DISCONNECTED',
                                         font=('Segoe UI', 10, 'bold'),
                                         bg='#0d1117', fg='#f85149')
        self.status_indicator.pack(side='right', padx=10)

        control_frame = ttk.LabelFrame(main_frame, text='Control Panel', padding='10')
        control_frame.grid(row=1, column=0, sticky='ew', pady=(0, 15))

        conn_section = ttk.Frame(control_frame)
        conn_section.pack(fill='x', pady=(0, 10))

        ttk.Label(conn_section, text="Host:").pack(side='left', padx=(0, 5))
        self.host_var = tk.StringVar(value='127.0.0.1')
        host_entry = ttk.Entry(conn_section, textvariable=self.host_var, width=12)
        host_entry.pack(side='left', padx=(0, 15))

        ttk.Label(conn_section, text="Port:").pack(side='left', padx=(0, 5))
        self.port_var = tk.StringVar(value='7497')
        port_entry = ttk.Entry(conn_section, textvariable=self.port_var, width=12)
        port_entry.pack(side='left', padx=(0, 15))

    
        ttk.Label(conn_section, text="Client ID:").pack(side='left', padx=(0, 5))
        self.client_id_var = tk.StringVar(value=str(self.client_id))
        client_id_entry = ttk.Entry(conn_section, textvariable=self.client_id_var, width=8)
        client_id_entry.pack(side='left', padx=(0, 15))
        
        self.connect_btn = ttk.Button(conn_section, text="Connect", command=self.connect_ib)
        self.connect_btn.pack(side='left', padx=(0, 5))
        self.disconnect_btn = ttk.Button(conn_section, text="Disconnect",
                                         command=self.disconnect_ib, state='disabled',
                                         style='Accent.TButton')
        self.disconnect_btn.pack(side='left')
        
        sep = ttk.Separator(control_frame, orient='horizontal')
        sep.pack(fill='x', pady=10)

        data_section = ttk.Frame(control_frame)
        data_section.pack(fill='x')

        ttk.Label(data_section, text='Symbol:').pack(side='left', padx=(0, 5))
        self.symbol_var = tk.StringVar(value='AAPL')
        symbol_entry = ttk.Entry(data_section, textvariable=self.symbol_var,
                                 width=10, font=('JetBrains Mono', 11))
        symbol_entry.pack(side='left', padx=(0, 5))

        self.stream_btn = ttk.Button(data_section, text="Start Stream",
                                     command=self.toggle_stream, state='disabled')
        self.stream_btn.pack(side='left', padx=(0, 5))

        self.recal_btn = ttk.Button(data_section, text="Recalibrate",
                                    command=self.recalibrate_model, state='disabled')
        self.recal_btn.pack(side='left', padx=(0, 15))

        price_frame = ttk.Frame(data_section)
        price_frame.pack(side='right')

        ttk.Label(price_frame, text="Last Price:",
                  font=('Segoe UI', 10)).pack(side='left', padx=(0, 5))
        self.price_label = tk.Label(price_frame, text='---.--',
                                    font=('JetBrains Mono', 16, 'bold'),
                                    bg='#0d1117', fg='#7ee787')
        self.price_label.pack(side='left')

        chart_frame = ttk.LabelFrame(main_frame, text='Live OHLC with Markov Regime (5s Bars)', padding=10)
        chart_frame.grid(row=2, column=0, sticky='nsew')
        chart_frame.columnconfigure(0, weight=1)
        chart_frame.rowconfigure(0, weight=1)
        
        self.chart_container = ttk.Frame(chart_frame)
        self.chart_container.grid(row=0, column=0, sticky='nsew')
        self.chart_container.columnconfigure(0, weight=1)
        self.chart_container.rowconfigure(0, weight=1)
        
        stats_frame = ttk.Frame(main_frame)
        stats_frame.grid(row=3, column=0, sticky='ew', pady=(10, 0))
        
        self.stats_labels = {}
        stats = [('Bars', '0'), ('High', '--'), ('Low', '--'),
                 ('Regime', '--'), ('Ticks/Bar', '0')]
        
        for i, (name, val) in enumerate(stats):
            frame = ttk.Frame(stats_frame)
            frame.pack(side='left', padx=15)
            ttk.Label(frame, text=f'{name}:', font=('Segoe UI', 9)).pack(side='left')
            label = tk.Label(frame, text=val, font=('JetBrains Mono', 10, 'bold'),
                             bg='#0d1117', fg='#8b949e')
            label.pack(side='left', padx=(5, 0))
            self.stats_labels[name] = label
    
    def setup_chart(self):
        plt.style.use('dark_background')
        
        self.fig, self.ax = plt.subplots(figsize=(12, 6), facecolor='#0d1117')
        self.ax.set_facecolor('#161b22')

        self.ax.tick_params(colors='#8b949e', labelsize=9)
        self.ax.spines['bottom'].set_color('#30363d')
        self.ax.spines['top'].set_color('#30363d')
        self.ax.spines['left'].set_color('#30363d')
        self.ax.spines['right'].set_color('#30363d')
        self.ax.grid(True, alpha=.2, color='#30363d', linestyle='--')
        
        self.ax.set_xlabel('Time', color='#8b949e', fontsize=10)
        self.ax.set_ylabel('Price', color='#8b949e', fontsize=10)
        self.ax.set_title('Waiting for data. . .', color='#c9d1d9', fontsize=12, fontweight='bold')

        self.canvas = FigureCanvasTkAgg(self.fig, self.chart_container)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')

        self.fig.tight_layout()
        self.canvas.draw()

    def create_contract(self, symbol):
        contract = Contract()
        contract.symbol = symbol.upper()
        contract.secType = 'STK'
        contract.exchange = 'SMART'
        contract.currency = 'USD'
        return contract

    def on_ib_error(self, error_code, error_string):
        """BUGFIX #5: surface contract/data errors to the user instead of
        letting the UI hang silently at 'Waiting for data...'. Called from
        the IB API's networking thread, so hop back to the main thread for
        any Tk calls."""
        def _show():
            messagebox.showerror(
                'Market Data Error',
                f"IB API error {error_code}: {error_string}"
            )
        
            if self.streaming:
                self.stop_stream()
        self.root.after(0, _show)

    def on_connection_error(self, error_code, error_string):
        """BUGFIX #7: connection-level errors (client id collision, not
        connected, etc) mean the session never actually came up, even if
        connect_ib's isConnected() check thought it did. Reset the UI back
        to a disconnected state and tell the user what happened, rather
        than silently trying to stream against a dead session."""
        def _show():
            was_connected = self.connected
            if self.streaming:
                self.stop_stream()
            self.connected = False
            self.connect_btn.config(state='normal')
            self.disconnect_btn.config(state='disabled')
            self.stream_btn.config(state='disabled')
            self.status_indicator.config(text='DISCONNECTED', fg='#f85149')

            if error_code == 326:
                messagebox.showerror(
                    'Connection Error',
                    "Client ID already in use (error 326). This usually "
                    "means a previous session wasn't released yet. Try "
                    "connecting again \u2014 a new random Client ID will be used."
                )
            elif was_connected or self.connecting:
                messagebox.showerror(
                    'Connection Error',
                    f"Lost connection to TWS/Gateway (error {error_code}): {error_string}"
                )
        self.root.after(0, _show)
    
    def connect_ib(self):
        
        if self.connecting or self.connected:
            return
        self.connecting = True
        self.connect_btn.config(state='disabled')

        try:
            host = self.host_var.get()
            port = int(self.port_var.get())

            
            try:
                self.client_id = int(self.client_id_var.get())
            except (ValueError, AttributeError):
                pass

          
            self.ib_app = IBAPP(callback=self.on_tick_data, error_callback=self.on_ib_error,
                                connection_error_callback=self.on_connection_error)

            def connect_thread():
                try:
                    print(f"Attempting to connect to {host}:{port} with clientId={self.client_id} ...")
                    self.ib_app.connect(host, port, clientId=self.client_id)
                    print(f"Socket connect() returned. isConnected()={self.ib_app.isConnected()}. Starting run() loop...")
                    self.ib_app.run()
                    print("run() loop exited.")
                except Exception as e:
                    import traceback
                    print(f"Connection error: {e}")
                    traceback.print_exc()

            thread = threading.Thread(target=connect_thread, daemon=True)
            thread.start()

           
            got_connected = self.ib_app.connected_event.wait(timeout=10)

            if got_connected:
                self.connected = True
                self.disconnect_btn.config(state='normal')
                self.stream_btn.config(state='normal')
                self.status_indicator.config(text='CONNECTED', fg='#7ee787')
            else:
                self.connect_btn.config(state='normal')
               
                self.client_id = random.randint(1000, 9999)
                self.client_id_var.set(str(self.client_id))
                messagebox.showerror('Error', 'Failed to connect to TWS. Verify API Port configurations.')
        except Exception as e:
            self.connect_btn.config(state='normal')
            messagebox.showerror('Error', f"Connection error: {e}")
        finally:
            self.connecting = False

    def disconnect_ib(self):
        try:
            if self.streaming:
                self.stop_stream()
            
            self.ib_app.disconnect()
            self.connected = False
            self.connect_btn.config(state='normal')
            self.disconnect_btn.config(state='disabled')
            self.stream_btn.config(state='disabled')
            self.status_indicator.config(text='DISCONNECTED', fg='#cb0000')
        except Exception as e:
            print(f"Disconnect error: {e}")

    def toggle_stream(self):
        if not self.streaming:
            self.start_stream()
        else:
            self.stop_stream()

    def start_stream(self):
        if not self.connected:
            return 
        
        symbol = self.symbol_var.get().upper()
        if not symbol:
            messagebox.showerror('Error', 'Please enter a symbol')
            return

        with self.bar_lock:
            self.ohlc_bars.clear()
            self.current_bar = None
            self.bar_start_time = None
            self.price_history.clear()
            self.regime_model = MarkovRegime()

        contract = self.create_contract(symbol)

        self.ib_app.historical_data.clear()
        self.ib_app.hist_done.clear()
        
        
        self.ib_app.reqMarketDataType(3)  
       
        self.ib_app.reqHistoricalData(2, contract, "", "300 S", "5 secs", "TRADES", 1, 1, False, [])

        if self.ib_app.hist_done.wait(timeout=10) and 2 in self.ib_app.historical_data:
            self.regime_model.calibrate(self.ib_app.historical_data[2])
            print(f"Calibrated regime model with {len(self.ib_app.historical_data[2])} bars")

        self.ib_app.reqMktData(1, contract, "", False, False, [])

        self.streaming = True
        self.running = True
        self.stream_btn.config(text='Stop Stream', style='Accent.TButton')
        self.recal_btn.config(state='normal')
        self.status_indicator.config(text=f'Streaming {symbol}', fg='#58a6ff')

        self.update_thread = threading.Thread(target=self.bar_manager_loop, daemon=True)
        self.update_thread.start()

        self.update_chart_loop()

    def stop_stream(self):
        self.running = False
        self.streaming = False

        try: 
            self.ib_app.cancelMktData(1)
        except Exception as e:
            print(f"Error canceling market data request: {e}")

        self.stream_btn.config(text='Start Stream', style='TButton')
        self.recal_btn.config(state='disabled')
        self.status_indicator.config(text='CONNECTED', fg='#7ee787')

    def recalibrate_model(self):
        if not self.streaming: 
            return 
        contract = self.create_contract(self.symbol_var.get().upper())
        self.ib_app.historical_data.clear()
        self.ib_app.hist_done.clear()

        
        self.ib_app.reqMarketDataType(3)
        

        
        self.ib_app.reqHistoricalData(3, contract, "", "300 S", "5 secs", "TRADES", 1, 1, False, [])
        if self.ib_app.hist_done.wait(timeout=10) and 3 in self.ib_app.historical_data: 
            with self.bar_lock:
                self.regime_model.calibrate(self.ib_app.historical_data[3])
            print(f"Recalibrated the Markov chain model with {len(self.ib_app.historical_data[3])} bars")

    def on_tick_data(self, data_type, value, timestamp):
        if data_type == 'price' and value > 0:
            with self.bar_lock:
                self.price_history.append((timestamp, value))

                if self.current_bar is None:
                    self.current_bar = OHLCBar(timestamp, value)
                    self.bar_start_time = timestamp
                else:
                    self.current_bar.update(value)

            self.root.after(0, lambda: self.price_label.config(text=f'{value:.2f}'))

    def bar_manager_loop(self):
        while self.running:
            time.sleep(.1)

            with self.bar_lock:
                if self.current_bar is not None and self.bar_start_time is not None:
                    elapsed = (datetime.now() - self.bar_start_time).total_seconds()

                    if elapsed >= self.bar_duration:
                        self.ohlc_bars.append(self.current_bar)
                        self.regime_model.get_regime(list(self.ohlc_bars))
                        last_price = self.current_bar.close
                        self.current_bar = OHLCBar(datetime.now(), last_price)
                        self.bar_start_time = datetime.now()

    def update_chart_loop(self):
        if not self.running:
            return
        self.draw_ohlc_chart()
        self.update_stats()
        self._after_id = self.root.after(200, self.update_chart_loop)

    def draw_ohlc_chart(self):
        self.ax.clear()

      
        with self.bar_lock:
            bars = list(self.ohlc_bars)
            current = self.current_bar
            current_regime_state = self.regime_model.current_state
            if current is not None:
                current.regime = current_regime_state

        if current is not None:
            bars = bars + [current]

        if not bars:
            self.ax.set_facecolor('#161b22')
            self.ax.set_title('Waiting for data. . .', color='#c9d1d9', fontsize=12, fontweight='bold')
            self.ax.grid(True, alpha=.2, color='#30363d', linestyle='--')
            return 
        
        all_prices = [bar.low for bar in bars] + [bar.high for bar in bars]
        price_min, price_max = min(all_prices), max(all_prices)
        price_range = price_max - price_min
        padding = max(price_range * .1, 0.01) 
        y_min, y_max = price_min - padding, price_max + padding

        width = .6

        for i, bar in enumerate(bars):
            bg = Rectangle(
                (i - .5, y_min),
                1, 
                y_max - y_min,
                facecolor=self.regime_model.bg_colors[bar.regime], 
                edgecolor='none',
                alpha=0.4,
                zorder=0
            )

            self.ax.add_patch(bg)

            color, edge_color = ('#3fb950', '#7ee787') if bar.close >= bar.open else ('#f85149', '#ff7b72')
            body_bottom, body_height = min(bar.open, bar.close), max(abs(bar.close - bar.open), .001)

            rect = Rectangle(
                (i - width/2, body_bottom),
                width,
                body_height,
                facecolor=color,
                edgecolor=edge_color,
                linewidth=1.5,
                alpha=.9,
                zorder=2
            )
            self.ax.add_patch(rect)

            self.ax.plot([i, i], [bar.low, body_bottom], color=edge_color, linewidth=1.5, zorder=1)
            self.ax.plot([i, i], [body_bottom + body_height, bar.high], color=edge_color, linewidth=1.5, zorder=1)

            if i == len(bars) - 1 and current is not None:
                self.ax.axvline(x=i, color='#58a6ff', alpha=.3, linestyle=':', linewidth=2)

        self.ax.set_facecolor('#161b22')
        x_labels = [bar.timestamp.strftime('%H:%M:%S') for bar in bars]
        self.ax.set_xticks(range(len(bars)))
        self.ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
        self.ax.set_ylim(y_min, y_max)
        self.ax.set_xlim(-.5, max(self.max_bars - .5, len(bars) - .5))
        self.ax.grid(True, alpha=.2, color='#30363d', linestyle='--')

        self.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
        self.ax.set_xlabel('Time', color='#8b949e', fontsize=10)
        self.ax.set_ylabel('Price', color='#8b949e', fontsize=10)

        symbol = self.symbol_var.get().upper()
        regime_names = ['LOW', 'MED', 'HIGH']
        curr_regime = regime_names[bars[-1].regime] if bars else 'N/A'
        self.ax.set_title(f'{symbol} - Regime: {curr_regime} | {len(bars)}/{self.max_bars} bars',
                          color='#c9d1d9', fontsize=12, fontweight='bold')
        
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def update_stats(self):
        with self.bar_lock:
            bars = list(self.ohlc_bars)
            current = self.current_bar

        if current:
            bars = bars + [current]

        if not bars:
            return 
        
        self.stats_labels['Bars'].config(text=str(len(bars)))

        all_highs = [b.high for b in bars]
        all_lows = [b.low for b in bars]
        self.stats_labels['High'].config(text=f'{max(all_highs):.2f}')
        self.stats_labels['Low'].config(text=f'{min(all_lows):.2f}')

        regime_names = ['LOW', 'MED', 'HIGH']
        regime_colors = ['#3fb950', '#d29922', '#f85149']
        curr_regime = bars[-1].regime if bars else 0 
        self.stats_labels['Regime'].config(text=regime_names[curr_regime], fg=regime_colors[curr_regime])

        if current:
            self.stats_labels['Ticks/Bar'].config(text=str(current.tick_count))

    def on_closing(self):
        self.running = False

        if hasattr(self, '_after_id'):
            self.root.after_cancel(self._after_id)

        if self.connected:
            try:
                if self.streaming:
                    self.ib_app.cancelMktData(1)
                self.ib_app.disconnect()
            except Exception as e:
                print(f"Error closing the application: {e}")
        
        self.root.destroy()

def main():
    root = tk.Tk()
    app = LiveMarketDashboard(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
