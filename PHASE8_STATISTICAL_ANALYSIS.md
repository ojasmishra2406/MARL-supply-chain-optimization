# Phase 8: Statistical Analysis & Experimental Validation

Date: 2026-10-01T12:17:14.723606


## 1. Experimental Setup

- **Algorithms Evaluated:** OUT (Baseline), IPPO, MAPPO, IPPO_COMM, MAPPO_COMM

- **Conditions:** baseline, high_variance

- **Independent Observations (Train Seeds):** up to 5 per configuration

- **Evaluation Seeds:** 5 per model per scenario

## 2. Descriptive Statistics (Aggregated across Evaluation Seeds)

| algorithm   | condition     | scenario            |   n_seeds |   cost_mean |   cost_std |   fill_rate_mean |   fill_rate_std |
|:------------|:--------------|:--------------------|----------:|------------:|-----------:|-----------------:|----------------:|
| ippo        | baseline      | capacity_disruption |         5 |    239611   |  31316.7   |         0.889487 |     0           |
| ippo        | baseline      | combined_shift      |         5 |     87149.6 |   4861.01  |         0.884506 |     0.0267532   |
| ippo        | baseline      | demand_shift        |         5 |     65888.8 |   4280.18  |         0.878162 |     0.0617674   |
| ippo        | baseline      | demand_spike        |         5 |     44224   |   4047.37  |         0.966195 |     0.0298316   |
| ippo        | baseline      | in_distribution     |         5 |     28184.2 |   1572.72  |         0.997332 |     0.000918256 |
| ippo        | baseline      | lead_time_shift     |         5 |     62430.5 |   3436.29  |         0.908491 |     0.0137858   |
| ippo        | high_variance | capacity_disruption |         5 |    228543   |  23665.6   |         0.889487 |     0           |
| ippo        | high_variance | combined_shift      |         5 |     89188.3 |   9072.02  |         0.905306 |     0.078596    |
| ippo        | high_variance | demand_shift        |         5 |     68785.4 |   6242.24  |         0.873363 |     0.0658124   |
| ippo        | high_variance | demand_spike        |         5 |     46406.1 |   3159.42  |         0.979864 |     0.0234648   |
| ippo        | high_variance | in_distribution     |         5 |     28508.4 |   1112.42  |         0.995614 |     0.00239183  |
| ippo        | high_variance | lead_time_shift     |         5 |     63959.6 |   5812.89  |         0.906887 |     0.0312191   |
| ippo_comm   | baseline      | capacity_disruption |         5 |    227975   |  24391.6   |         0.889487 |     0           |
| ippo_comm   | baseline      | combined_shift      |         5 |     84431.2 |   6344.3   |         0.887291 |     0.0799592   |
| ippo_comm   | baseline      | demand_shift        |         5 |     64778.4 |   6997.9   |         0.813491 |     0.108478    |
| ippo_comm   | baseline      | demand_spike        |         5 |     40800.4 |   2939.51  |         0.967927 |     0.0170128   |
| ippo_comm   | baseline      | in_distribution     |         5 |     28022   |    784.765 |         0.993994 |     0.00242934  |
| ippo_comm   | baseline      | lead_time_shift     |         5 |     60998   |   5124.33  |         0.909524 |     0.0386592   |
| ippo_comm   | high_variance | capacity_disruption |         5 |    224358   |  41833.4   |         0.889487 |     0           |
| ippo_comm   | high_variance | combined_shift      |         5 |     85794   |   4955.98  |         0.874668 |     0.0521699   |
| ippo_comm   | high_variance | demand_shift        |         5 |     67902.5 |   7959.15  |         0.882019 |     0.0262088   |
| ippo_comm   | high_variance | demand_spike        |         5 |     49533.4 |   9264.29  |         0.948618 |     0.0514321   |
| ippo_comm   | high_variance | in_distribution     |         5 |     28293.2 |   1012.35  |         0.995564 |     0.00191751  |
| ippo_comm   | high_variance | lead_time_shift     |         5 |     68487.9 |   5557.85  |         0.894639 |     0.0209979   |
| mappo       | baseline      | capacity_disruption |         5 |    260156   |  44137.8   |         0.847735 |     0.0933613   |
| mappo       | baseline      | combined_shift      |         5 |    102466   |   5587.2   |         0.8687   |     0.221442    |
| mappo       | baseline      | demand_shift        |         5 |     76965.7 |   6555.69  |         0.839688 |     0.180397    |
| mappo       | baseline      | demand_spike        |         5 |     41976.9 |   4491.82  |         0.965323 |     0.0255292   |
| mappo       | baseline      | in_distribution     |         5 |     32882.1 |   2841.11  |         0.997779 |     0.00165564  |
| mappo       | baseline      | lead_time_shift     |         5 |     65548.3 |   9600.8   |         0.945571 |     0.0327282   |
| mappo       | high_variance | capacity_disruption |         5 |    252070   |  45017.8   |         0.848341 |     0.0920056   |
| mappo       | high_variance | combined_shift      |         5 |    100837   |   4522.73  |         0.811837 |     0.207369    |
| mappo       | high_variance | demand_shift        |         5 |     76690   |   9274.62  |         0.759017 |     0.185736    |
| mappo       | high_variance | demand_spike        |         5 |     45113   |   3948.43  |         0.924316 |     0.0604148   |
| mappo       | high_variance | in_distribution     |         5 |     33581.1 |   1459.15  |         0.99513  |     0.00299273  |
| mappo       | high_variance | lead_time_shift     |         5 |     62555.9 |   2744.34  |         0.929762 |     0.0710071   |
| mappo_comm  | baseline      | capacity_disruption |         4 |    291594   |  23067.3   |         0.889487 |     0           |
| mappo_comm  | baseline      | combined_shift      |         4 |    108810   |  12915.6   |         0.945209 |     0.0310726   |
| mappo_comm  | baseline      | demand_shift        |         4 |     73990.7 |   5320.05  |         0.924101 |     0.0180591   |
| mappo_comm  | baseline      | demand_spike        |         4 |     56401.4 |   5160.85  |         0.86809  |     0.0327038   |
| mappo_comm  | baseline      | in_distribution     |         4 |     31684.9 |   2150.37  |         0.997829 |     0.00251032  |
| mappo_comm  | baseline      | lead_time_shift     |         4 |     63313.6 |   5651.38  |         0.957591 |     0.0411171   |
| mappo_comm  | high_variance | capacity_disruption |         5 |    236889   |  41118.1   |         0.886709 |     0.00621152  |
| mappo_comm  | high_variance | combined_shift      |         5 |     97556.4 |   5341.54  |         0.717431 |     0.172349    |
| mappo_comm  | high_variance | demand_shift        |         5 |     66738.2 |   8099.44  |         0.748143 |     0.176079    |
| mappo_comm  | high_variance | demand_spike        |         5 |     50805.9 |   4149.89  |         0.829546 |     0.074159    |
| mappo_comm  | high_variance | in_distribution     |         5 |     34280.6 |   1670.66  |         0.997643 |     0.0026787   |
| mappo_comm  | high_variance | lead_time_shift     |         5 |     63921.8 |   3526.84  |         0.923096 |     0.0573135   |
| out         | baseline      | capacity_disruption |         5 |     88425   |    516.263 |         0.889487 |     0.0161178   |
| out         | baseline      | combined_shift      |         5 |    161318   |   2289.87  |         1        |     0           |
| out         | baseline      | demand_shift        |         5 |    194909   |   3132.44  |         1        |     0           |
| out         | baseline      | demand_spike        |         5 |    176488   |    982.946 |         1        |     0           |
| out         | baseline      | in_distribution     |         5 |    180330   |   1000.3   |         1        |     0           |
| out         | baseline      | lead_time_shift     |         5 |    309018   |    921.579 |         1        |     0           |



## 3. Statistical Testing (Welch's t-test, BH-FDR Corrected)

| name                                                         |          p |         t |         d |   n1 |   n2 |    mean1 |    mean2 |     adj_p |
|:-------------------------------------------------------------|-----------:|----------:|----------:|-----:|-----:|---------:|---------:|----------:|
| IPPO vs MAPPO (Baseline, In-Distribution)                    | 0.0168416  | -3.23485  | -2.0459   |    5 |    5 |  28184.2 |  32882.1 | 0.0673663 |
| IPPO vs IPPO_COMM (Baseline, In-Distribution)                | 0.843424   |  0.206427 |  0.130556 |    5 |    5 |  28184.2 |  28022   | 0.843424  |
| MAPPO vs MAPPO_COMM (Baseline, In-Distribution)              | 0.495295   |  0.719264 |  0.466204 |    5 |    4 |  32882.1 |  31684.9 | 0.660394  |
| IPPO (Base) vs IPPO (High Var) (In-Distribution)             | 0.717556   | -0.376273 | -0.237976 |    5 |    5 |  28184.2 |  28508.4 | 0.820064  |
| MAPPO_COMM (Base) vs MAPPO_COMM (High Var) (In-Distribution) | 0.097967   | -1.98256  | -1.37253  |    4 |    5 |  31684.9 |  34280.6 | 0.261245  |
| IPPO vs MAPPO (Baseline, Combined Shift)                     | 0.00178698 | -4.62457  | -2.92484  |    5 |    5 |  87149.6 | 102466   | 0.0142959 |
| IPPO vs IPPO_COMM (Baseline, Combined Shift)                 | 0.470186   |  0.760537 |  0.481006 |    5 |    5 |  87149.6 |  84431.2 | 0.660394  |
| MAPPO vs MAPPO_COMM (Baseline, Combined Shift)               | 0.41263    | -0.916219 | -0.671245 |    5 |    4 | 102466   | 108810   | 0.660394  |



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
| mappo_comm  | baseline      |        31684.9 |       108810   | 243.41%       |
| mappo_comm  | high_variance |        34280.6 |        97556.4 | 184.58%       |
| out         | baseline      |       180330   |       161318   | -10.54%       |


