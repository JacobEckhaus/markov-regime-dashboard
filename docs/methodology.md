# Methodology

This document explains the quantitative methodology used in the Markov Regime-Switching Market Dashboard.

## Regime Definition

The model represents the market using three discrete volatility regimes:

1. Low volatility
2. Medium volatility
3. High volatility

Each incoming 5-second OHLC bar is converted into a relative volatility measure:

```text
Volatility = (High - Low) / Close
```

This creates a normalized measure of short-term price movement that can be compared across different price levels.

## Historical Calibration

Before live classification begins, the model retrieves historical 5-second OHLC data and uses those observations to calibrate the regime model.

Historical volatility observations are divided into three groups using percentile thresholds:

- Bottom third → Low volatility
- Middle third → Medium volatility
- Top third → High volatility

For each regime, the model estimates:

- Mean volatility
- Standard deviation of volatility
- Transition probabilities to other regimes

This allows the model parameters to adapt to the recent behavior of the selected security rather than relying entirely on fixed assumptions.

## Transition Matrix

The Markov model assumes that the probability of the next regime depends on the current regime.

The transition matrix is defined as:

```text
P[i, j] = Probability of moving from regime i to regime j
```

An example transition matrix is:

```text
        Low    Medium   High
Low     0.90    0.08    0.02
Medium  0.10    0.80    0.10
High    0.02    0.08    0.90
```

During calibration, empirical regime transitions observed in historical data are used to update these probabilities.

## Gaussian Emission Model

Each regime is modeled using a Gaussian distribution of observed volatility.

For regime `j`, the likelihood of observing volatility `x` is:

```text
Likelihood(x | regime j)
=
Gaussian(x; mean_j, std_j)
```

Low-volatility regimes are expected to have lower emission means, while high-volatility regimes are expected to have higher emission means.

## State Prediction

Before incorporating the newest volatility observation, the previous regime probabilities are propagated through the transition matrix:

```text
Predicted State Probabilities
=
Transition Matrixᵀ × Previous State Probabilities
```

This represents the model's prior belief about the next market regime.

## Bayesian Update

The predicted probabilities are combined with the Gaussian likelihood of the new volatility observation.

For each regime:

```text
Posterior Probability
∝
Predicted State Probability
×
Observation Likelihood
```

The posterior probabilities are then normalized so that:

```text
P(Low) + P(Medium) + P(High) = 1
```

The regime with the highest posterior probability is classified as the current market regime.

## Live Data Pipeline

The application receives streaming price updates through the Interactive Brokers API.

Incoming trades are aggregated into rolling 5-second OHLC bars:

```text
Tick Data
    ↓
5-Second OHLC Bar
    ↓
Relative Volatility
    ↓
Markov Prediction
    ↓
Gaussian Likelihood
    ↓
Bayesian Posterior Update
    ↓
Regime Classification
```

The resulting classification is displayed on the dashboard alongside the live candlestick chart.

## Dynamic Recalibration

The model can be recalibrated using more recent historical data.

This updates:

- Regime volatility distributions
- Transition probabilities
- State behavior

Recalibration allows the model to adjust when the statistical characteristics of the underlying security change over time.

## Limitations

This implementation is a simplified regime-classification model and has several limitations:

- It uses short-term high-low volatility as the primary observation variable.
- Regimes are defined using historical percentile thresholds.
- Gaussian emissions may not fully capture heavy-tailed financial return distributions.
- The model does not currently incorporate volume, realized volatility over longer horizons, macroeconomic variables, or additional market features.
- The detected regimes are descriptive market states and are not by themselves trading signals.

Future improvements could include hidden Markov model estimation, additional features, regime-conditioned return analysis, and out-of-sample validation.
