# Phase 9: Demand Forecasting Pipeline

Date: 2026-09-23T15:56:30.764282


## 1. Experimental Setup

- **Training Data**: Historical simulator demand (Temporal Split)

- **History Window**: 5

- **Forecast Horizon**: 1

- **Models Evaluated**: Naive, Moving Average, Exponential Smoothing, XGBoost


## 2. Metrics (In-Distribution / Validation)

| Model         |     MAE |    RMSE |    sMAPE |       Bias |
|:--------------|--------:|--------:|---------:|-----------:|
| Naive         | 4.51064 | 5.37528 | 23.0918  | 0.0425532  |
| MovingAverage | 3.47234 | 4.19949 | 17.9424  | 0.0765957  |
| ExpSmoothing  | 3.42094 | 4.16481 | 17.6524  | 0.121655   |
| XGBoost       | 1.27778 | 1.56666 |  6.89327 | 0.00543623 |



## 3. Metrics (Out-of-Distribution / Test)

| Model         |      MAE |    RMSE |   sMAPE |       Bias |
|:--------------|---------:|--------:|--------:|-----------:|
| Naive         | 11.4681  | 22.9008 | 35.3976 |  0.0638298 |
| MovingAverage |  9.37447 | 18.2833 | 30.3149 |  0.0978723 |
| ExpSmoothing  |  9.03161 | 18.0555 | 29.2537 |  0.112863  |
| XGBoost       |  8.48142 | 18.4274 | 29.1589 | -5.2734    |



## 4. Integration

The MARL observation interface (`forecasting.wrapper.DemandForecastWrapper`) has been successfully tested. 
It deterministically injects historical predictions into the observation space without leaking future data.
