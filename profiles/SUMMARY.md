| comparison | pairs | min self-repeat top1 (ref/cand) | max self spread | min cross top1 | median cross top1 | p99 |logprob Δ| max over prompts | max |Δ| | completion exact match | first divergence steps | warnings |
|---|---|---|---|---|---|---|---|---|---|---|
| tp1_bf16_vs_fp8 | 16 | 1.000/1.000 | 0.0000 | 0.800 | 0.952 | 1.594 | 1.737 | 7/16 | [17, 10, 5, 33, 0, 38, 15, 9, 11] | none |
| tp2_bf16_vs_fp8 | 16 | 1.000/1.000 | 0.0000 | 0.800 | 0.938 | 1.315 | 1.408 | 6/16 | [10, 41, 27, 5, 25, 0, 44, 20, 9, 17] | none |
| fp8_tp1_vs_tp2 | 16 | 1.000/1.000 | 0.0000 | 0.800 | 1.000 | 1.069 | 1.151 | 7/16 | [17, 10, 41, 27, 25, 35, 38, 15, 11] | accelerator set differs; tensor_parallel_size differs |
| bf16_tp1_vs_tp2 | 16 | 1.000/1.000 | 0.0000 | 0.923 | 1.000 | 0.429 | 0.459 | 14/16 | [32, 55] | accelerator set differs; tensor_parallel_size differs |

### fp8_tp1_vs_tp2 per prompt (label, tokens, cross top1, p99 |Δ|, max |Δ|, exact completion)
p00 14 1.000 1.069 1.151 True div=None
p01 13 1.000 0.448 0.478 False div=17
p02 25 0.958 0.240 0.243 True div=None
p03 12 1.000 0.253 0.255 False div=10
p04 24 0.957 0.331 0.381 True div=None
p05 22 1.000 0.170 0.179 False div=41
p06 22 0.857 0.308 0.328 True div=None
p07 23 1.000 0.564 0.615 False div=27
p08 14 1.000 0.337 0.341 True div=None
p09 10 1.000 0.613 0.636 False div=25
p10 13 1.000 0.287 0.292 True div=None
p11 14 0.923 0.164 0.167 False div=35
p12 10 1.000 0.374 0.384 False div=38
p13 11 0.800 0.423 0.446 False div=15
p14 22 0.905 0.266 0.285 True div=None
p15 27 0.923 0.474 0.478 False div=11

### bf16_tp1_vs_tp2 per prompt (label, tokens, cross top1, p99 |Δ|, max |Δ|, exact completion)
p00 14 1.000 0.416 0.459 True div=None
p01 13 1.000 0.070 0.073 True div=None
p02 25 1.000 0.167 0.181 True div=None
p03 12 1.000 0.176 0.181 True div=None
p04 24 1.000 0.155 0.184 True div=None
p05 22 0.952 0.102 0.104 True div=None
p06 22 1.000 0.113 0.118 True div=None
p07 23 1.000 0.122 0.122 True div=None
p08 14 1.000 0.124 0.131 True div=None
p09 10 1.000 0.108 0.109 True div=None
p10 13 1.000 0.086 0.088 True div=None
p11 14 0.923 0.097 0.100 True div=None
p12 10 1.000 0.429 0.459 True div=None
p13 11 1.000 0.112 0.117 False div=32
p14 22 1.000 0.141 0.156 True div=None
p15 27 1.000 0.236 0.256 False div=55

### tp1_bf16_vs_fp8 per prompt (label, tokens, cross top1, p99 |Δ|, max |Δ|, exact completion)
p00 14 0.846 1.049 1.075 True div=None
p01 13 1.000 0.252 0.261 False div=17
p02 25 0.875 0.389 0.412 True div=None
p03 12 0.818 0.376 0.381 False div=10
p04 24 1.000 1.020 1.257 True div=None
p05 22 0.952 0.941 1.149 True div=None
p06 22 0.952 0.852 0.964 True div=None
p07 23 1.000 1.400 1.476 True div=None
p08 14 1.000 0.461 0.463 False div=5
p09 10 1.000 1.594 1.670 False div=33
p10 13 0.917 0.603 0.651 True div=None
p11 14 0.923 0.485 0.510 False div=0
p12 10 1.000 0.270 0.276 False div=38
p13 11 0.800 0.607 0.630 False div=15
p14 22 0.952 0.646 0.744 False div=9
p15 27 0.962 1.511 1.737 False div=11

### tp2_bf16_vs_fp8 per prompt (label, tokens, cross top1, p99 |Δ|, max |Δ|, exact completion)
p00 14 0.846 0.791 0.870 True div=None
p01 13 1.000 0.341 0.366 True div=None
p02 25 0.917 0.275 0.279 True div=None
p03 12 0.818 0.513 0.515 False div=10
p04 24 0.957 0.803 0.921 True div=None
p05 22 1.000 1.009 1.225 False div=41
p06 22 0.905 0.762 0.858 True div=None
p07 23 1.000 0.881 0.983 False div=27
p08 14 1.000 0.356 0.363 False div=5
p09 10 1.000 1.315 1.408 False div=25
p10 13 0.917 0.363 0.366 True div=None
p11 14 0.923 0.534 0.580 False div=0
p12 10 1.000 0.383 0.396 False div=44
p13 11 0.800 0.523 0.528 False div=20
p14 22 0.952 0.614 0.682 False div=9
p15 27 0.885 1.172 1.238 False div=17

