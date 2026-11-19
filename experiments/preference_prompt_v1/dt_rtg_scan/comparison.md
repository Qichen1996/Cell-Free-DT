# Pure DT RTG sweep versus preference PDT

Same held-out seeds6–10,7days,model fixed,common evaluation rewards40/.5/.5/3. Baseline reused,20new simulations.

| Model | Actual RTG | Preference pc | Power W | Stringent drop % | Sensitive drop % | Tolerant drop % |
|---|---:|---:|---:|---:|---:|---:|
| DT | -535.276988354 | — | 292.4129 | 0.240838 | 0.027126 | 0.009434 |
| DT | -485.276988354 | — | 293.1447 | 0.219895 | 0.028440 | 0.010781 |
| DT | -435.276988354 | — | 292.8075 | 0.115183 | 0.021733 | 0.009796 |
| DT | -385.276988354 | — | 294.5335 | 0.188482 | 0.023336 | 0.009345 |
| DT | -335.276988354 | — | 296.6734 | 0.146597 | 0.026252 | 0.008537 |
| PDT | -435.276988354 | 0.6 | 321.9623 | 0.010471 | 0.007583 | 0.002246 |
| PDT | -435.276988354 | 1.5 | 307.8696 | 0.020942 | 0.036756 | 0.012761 |
| PDT | -435.276988354 | 1.8 | 295.7838 | 0.083770 | 0.010646 | 0.007547 |
| PDT | -435.276988354 | 2.1 | 290.3310 | 0.136126 | 0.028440 | 0.008176 |

Finite sweep: lack of matching point is not proof that RTG cannot substitute for prompt. No extra automatic sweep.