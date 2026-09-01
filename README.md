# Markov Regime-Switching Market Dashboard

A live market regime detection and visualization system built in Python using Interactive Brokers market data and a 3-state Markov model.

## Dashboard Preview

> Screenshot coming soon

## Overview

This project classifies changing market conditions into three volatility regimes:

- Low volatility
- Medium volatility
- High volatility

The application receives streaming equity market data through the Interactive Brokers API, aggregates incoming price data into rolling OHLC bars, calculates volatility, and updates regime probabilities using a Markov model.

A multithreaded desktop dashboard visualizes candlestick data and highlights the detected volatility regime as market conditions change.


## Features

- Interactive Brokers API market-data integration
- Rolling 5-second OHLC bar construction
- Three-state volatility regime classification
- Historically calibrated transition probabilities
- Gaussian emission distributions
- Bayesian posterior probability updates
- Live candlestick visualization
- Regime-based chart highlighting
- Dynamic model recalibration
- Multithreaded data processing and GUI updates

## Model

The model represents the market using three latent volatility states:

1. Low Volatility
2. Medium Volatility
3. High Volatility

For each new observation, the previous state probabilities are propagated through the transition matrix:

```text
Predicted State Probabilities = Pᵀ × Previous State Probabilities
```

The model then evaluates the observed volatility under the Gaussian emission distribution associated with each state.

The posterior regime probabilities are proportional to:

```text
Posterior ∝ Predicted State Probability × Observation Likelihood
```

The probabilities are normalized, and the state with the highest posterior probability is classified as the current market regime.

## Calibration

Historical 5-second OHLC bars are used to calibrate the model.

Observed volatility is calculated using the relative high-low range of each bar. Historical observations are divided into low-, medium-, and high-volatility groups using percentile thresholds.

The calibration process estimates:

- Mean volatility for each regime
- Standard deviation for each regime
- Empirical transition probabilities between regimes

## Tech Stack

- Python
- NumPy
- Matplotlib
- Tkinter
- Interactive Brokers API

## Project Structure

```text
markov-regime-dashboard/
├── src/
│   └── markov_dashboard.py
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Requirements

This project requires Interactive Brokers Trader Workstation (TWS) or IB Gateway with API access enabled.

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

## Disclaimer

This project is for educational and research purposes only and is not intended to provide financial or investment advice.
