# Phase 8: Statistical Analysis & Experimental Validation

Date: 2026-09-23T15:55:26.656643


## 1. Experimental Setup

- **Algorithms Evaluated:** OUT (Baseline), IPPO, MAPPO, IPPO_COMM, MAPPO_COMM

- **Conditions:** baseline, high_variance

- **Independent Observations (Train Seeds):** up to 5 per configuration

- **Evaluation Seeds:** 5 per model per scenario

## 2. Descriptive Statistics (Aggregated across Evaluation Seeds)

| algorithm   | condition     | scenario            |   n_seeds |   cost_mean |   cost_std |   fill_rate_mean |   fill_rate_std |
|:------------|:--------------|:--------------------|----------:|------------:|-----------:|-----------------:|----------------:|
| ippo        | baseline      | capacity_disruption |         5 |    239611   |    816.192 |         0.889487 |      0.0161178  |
| ippo        | baseline      | combined_shift      |         5 |     87149.6 |   2261.98  |         0.884506 |      0.0219579  |
| ippo        | baseline      | demand_shift        |         5 |     65888.8 |   1411.63  |         0.878162 |      0.0218603  |
| ippo        | baseline      | demand_spike        |         5 |     44224   |    671.141 |         0.966195 |      0.0136116  |
| ippo        | baseline      | in_distribution     |         5 |     28184.2 |    376.157 |         0.997332 |      0.00543421 |
| ippo        | baseline      | lead_time_shift     |         5 |     62430.5 |    888.352 |         0.908491 |      0.0201763  |
| ippo        | high_variance | capacity_disruption |         5 |    228543   |    688.057 |         0.889487 |      0.0161178  |
| ippo        | high_variance | combined_shift      |         5 |     89188.3 |   1648.17  |         0.905306 |      0.0191222  |
| ippo        | high_variance | demand_shift        |         5 |     68785.4 |   1180.57  |         0.873363 |      0.0205389  |
| ippo        | high_variance | demand_spike        |         5 |     46406.1 |    644.133 |         0.979864 |      0.0103841  |
| ippo        | high_variance | in_distribution     |         5 |     28508.4 |    568.548 |         0.995614 |      0.0069889  |
| ippo        | high_variance | lead_time_shift     |         5 |     63959.6 |    891.605 |         0.906887 |      0.019895   |
| ippo_comm   | baseline      | capacity_disruption |         5 |    227975   |    898.14  |         0.889487 |      0.0161178  |
| ippo_comm   | baseline      | combined_shift      |         5 |     84431.2 |   1481.33  |         0.887291 |      0.0221675  |
| ippo_comm   | baseline      | demand_shift        |         5 |     64778.4 |   1706.52  |         0.813491 |      0.0180805  |
| ippo_comm   | baseline      | demand_spike        |         5 |     40800.4 |    687.011 |         0.967927 |      0.0190131  |
| ippo_comm   | baseline      | in_distribution     |         5 |     28022   |    414.456 |         0.993994 |      0.00989437 |
| ippo_comm   | baseline      | lead_time_shift     |         5 |     60998   |    574.037 |         0.909524 |      0.0205103  |
| ippo_comm   | high_variance | capacity_disruption |         5 |    224358   |    425.946 |         0.889487 |      0.0161178  |
| ippo_comm   | high_variance | combined_shift      |         5 |     85794   |   1156.96  |         0.874668 |      0.0213864  |
| ippo_comm   | high_variance | demand_shift        |         5 |     67902.5 |    890.83  |         0.882019 |      0.0221096  |
| ippo_comm   | high_variance | demand_spike        |         5 |     49533.4 |    685.476 |         0.948618 |      0.0125917  |
| ippo_comm   | high_variance | in_distribution     |         5 |     28293.2 |    669.567 |         0.995564 |      0.00896359 |
| ippo_comm   | high_variance | lead_time_shift     |         5 |     68487.9 |   1273.5   |         0.894639 |      0.0217558  |
| mappo       | baseline      | capacity_disruption |         5 |    260156   |   2903.62  |         0.847735 |      0.0142663  |
| mappo       | baseline      | combined_shift      |         5 |    102466   |   3569.43  |         0.8687   |      0.0166954  |
| mappo       | baseline      | demand_shift        |         5 |     76965.7 |   2681.06  |         0.839688 |      0.0136957  |
| mappo       | baseline      | demand_spike        |         5 |     41976.9 |   1245.24  |         0.965323 |      0.0224293  |
| mappo       | baseline      | in_distribution     |         5 |     32882.1 |    354.601 |         0.997779 |      0.00400748 |
| mappo       | baseline      | lead_time_shift     |         5 |     65548.3 |   1477.99  |         0.945571 |      0.0176175  |
| mappo       | high_variance | capacity_disruption |         5 |    252070   |   1355.24  |         0.848341 |      0.016139   |
| mappo       | high_variance | combined_shift      |         5 |    100837   |   3297.96  |         0.811837 |      0.0193541  |
| mappo       | high_variance | demand_shift        |         5 |     76690   |   2438.53  |         0.759017 |      0.0130244  |
| mappo       | high_variance | demand_spike        |         5 |     45113   |    925.982 |         0.924316 |      0.0218373  |
| mappo       | high_variance | in_distribution     |         5 |     33581.1 |    514.194 |         0.99513  |      0.00625533 |
| mappo       | high_variance | lead_time_shift     |         5 |     62555.9 |    480.453 |         0.929762 |      0.0170935  |
| mappo_comm  | baseline      | capacity_disruption |         5 |    302358   |   2312.26  |         0.889487 |      0.0161178  |
| mappo_comm  | baseline      | combined_shift      |         5 |    114288   |   4435.7   |         0.95345  |      0.0263273  |
| mappo_comm  | baseline      | demand_shift        |         5 |     76096.6 |   2315.42  |         0.927835 |      0.0284802  |
| mappo_comm  | baseline      | demand_spike        |         5 |     58880.5 |   1341.17  |         0.869572 |      0.0103882  |
| mappo_comm  | baseline      | in_distribution     |         5 |     32284.7 |    541.448 |         0.997105 |      0.00507565 |
| mappo_comm  | baseline      | lead_time_shift     |         5 |     65595.5 |   1839.06  |         0.951377 |      0.0203309  |
| mappo_comm  | high_variance | capacity_disruption |         5 |    236889   |    300.747 |         0.886709 |      0.0157406  |
| mappo_comm  | high_variance | combined_shift      |         5 |     97556.4 |   1307.04  |         0.717431 |      0.0155632  |
| mappo_comm  | high_variance | demand_shift        |         5 |     66738.2 |   1489.42  |         0.748143 |      0.01366    |
| mappo_comm  | high_variance | demand_spike        |         5 |     50805.9 |    898.054 |         0.829546 |      0.0133624  |
| mappo_comm  | high_variance | in_distribution     |         5 |     34280.6 |    303.354 |         0.997643 |      0.00210565 |
| mappo_comm  | high_variance | lead_time_shift     |         5 |     63921.8 |   1499.26  |         0.923096 |      0.0191206  |
| out         | baseline      | capacity_disruption |         5 |     88425   |    516.263 |         0.889487 |      0.0161178  |
| out         | baseline      | combined_shift      |         5 |    161318   |   2289.87  |         1        |      0          |
| out         | baseline      | demand_shift        |         5 |    194909   |   3132.44  |         1        |      0          |
| out         | baseline      | demand_spike        |         5 |    176488   |    982.946 |         1        |      0          |
| out         | baseline      | in_distribution     |         5 |    180330   |   1000.3   |         1        |      0          |
| out         | baseline      | lead_time_shift     |         5 |    309018   |    921.579 |         1        |      0          |



## 3. Statistical Testing (Welch's t-test, BH-FDR Corrected)

| name                                                         |           p |          t |          d |   n1 |   n2 |    mean1 |    mean2 |       adj_p |
|:-------------------------------------------------------------|------------:|-----------:|-----------:|-----:|-----:|---------:|---------:|------------:|
| IPPO vs MAPPO (Baseline, In-Distribution)                    | 3.75274e-08 | -20.3207   | -12.8519   |    5 |    5 |  28184.2 |  32882.1 | 3.00219e-07 |
| IPPO vs IPPO_COMM (Baseline, In-Distribution)                | 0.535151    |   0.648244 |   0.409985 |    5 |    5 |  28184.2 |  28022   | 0.535151    |
| MAPPO vs MAPPO_COMM (Baseline, In-Distribution)              | 0.0785267   |   2.06373  |   1.30522  |    5 |    5 |  32882.1 |  32284.7 | 0.104702    |
| IPPO (Base) vs IPPO (High Var) (In-Distribution)             | 0.323262    |  -1.06326  |  -0.672464 |    5 |    5 |  28184.2 |  28508.4 | 0.369442    |
| MAPPO_COMM (Base) vs MAPPO_COMM (High Var) (In-Distribution) | 0.000295945 |  -7.19098  |  -4.54797  |    5 |    5 |  32284.7 |  34280.6 | 0.000789188 |
| IPPO vs MAPPO (Baseline, Combined Shift)                     | 0.000100777 |  -8.10469  |  -5.12586  |    5 |    5 |  87149.6 | 102466   | 0.000403108 |
| IPPO vs IPPO_COMM (Baseline, Combined Shift)                 | 0.0599154   |   2.24811  |   1.42183  |    5 |    5 |  87149.6 |  84431.2 | 0.0958647   |
| MAPPO vs MAPPO_COMM (Baseline, Combined Shift)               | 0.00186821  |  -4.6431   |  -2.93655  |    5 |    5 | 102466   | 114288   | 0.00373642  |



## 4. Distribution-Shift Robustness Analysis

Comparing relative degradation from `in_distribution` to `combined_shift`.


| Algorithm   | Condition     |   In-Dist Cost |   Shifted Cost | Degradation   |
|:------------|:--------------|---------------:|---------------:|:--------------|
| ippo        | baseline      |        28184.2 |        87149.6 | 209.21%       |
| ippo        | high_variance |        28508.4 |        89188.3 | 212.85%       |
| ippo_comm   | baseline      |        28022   |        84431.2 | 201.30%       |
| ippo_comm   | high_variance |        28293.2 |        85794   | 203.23%       |
| mappo       | baseline      |        32882.1 |       102466   | 211.62%       |
| mappo       | high_variance |        33581.1 |       100837   | 200.28%       |
| mappo_comm  | baseline      |        32284.7 |       114288   | 254.00%       |
| mappo_comm  | high_variance |        34280.6 |        97556.4 | 184.58%       |
| out         | baseline      |       180330   |       161318   | -10.54%       |


