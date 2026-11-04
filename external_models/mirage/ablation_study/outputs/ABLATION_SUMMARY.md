# Mirage ablation summary

Source: `outputs` · reference per benchmark: `target__K4__none__forced` · paired bootstrap B=10000 (seed 20261103) · equivalence margin ±3.0 pts (90% CI) · McNemar exact two-sided; `p (Holm)` = Holm over all conditions of the benchmark, `p (Holm, family)` = Holm over the confirmatory family `target__K4__zero__forced,target__K4__noise__forced,target__K4__donor__forced,target__K4__pause__forced,target__K4__mean__forced,target__K0__none__forced,target__Kauto__suppress__natural`.

## blink

| Condition | n | Acc (%) | SE | 95% Wilson | Δ vs ref (pts) | 95% bootstrap CI | McNemar b/c | p | p (Holm) | Equiv. | Latent steps | ‖h‖ | cos(h_t,h_t-1) | max cos(h,img) | Unparsed |
|---|---:|---:|---:|---|---:|---|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| base · K=0 · none · natural | 150 | 74.00 | 3.58 | [66.44, 80.36] | +1.33 | [-5.33, +8.00] | 13/15 | 0.8506 | 1.0000 | no | 0.00 | — | — | — | 0.00 |
| target · K=auto · none · natural | 150 | 73.33 | 3.61 | [65.74, 79.76] | +0.67 | [-2.67, +4.00] | 3/4 | 1.0000 | 1.0000 | no | 0.00 | — | — | — | 0.00 |
| target · K=0 · none · forced | 150 | 70.00 | 3.74 | [62.24, 76.76] | -2.67 | [-6.00, +0.00] | 5/1 | 0.2188 | 1.0000 | no | 0.00 | — | — | — | 0.00 |
| target · K=1 · none · forced | 150 | 70.67 | 3.72 | [62.94, 77.36] | -2.00 | [-4.67, +0.67] | 4/1 | 0.3750 | 1.0000 | no | 1.00 | 176.2 | — | 0.161 | 0.00 |
| target · K=3 · none · forced | 150 | 72.67 | 3.64 | [65.04, 79.17] | +0.00 | [-2.00, +2.00] | 1/1 | 1.0000 | 1.0000 | yes | 3.00 | 171.2 | 0.832 | 0.159 | 0.00 |
| target · K=4 · none · forced **(ref)** | 150 | 72.67 | 3.64 | [65.04, 79.17] | — | — | — | — | — | — | 4.00 | 174.3 | 0.856 | 0.155 | 0.00 |
| target · K=5 · none · forced | 150 | 72.00 | 3.67 | [64.33, 78.57] | -0.67 | [-2.67, +1.33] | 2/1 | 1.0000 | 1.0000 | yes | 5.00 | 184.6 | 0.864 | 0.149 | 0.00 |
| target · K=10 · none · forced | 150 | 71.33 | 3.69 | [63.64, 77.97] | -1.33 | [-4.67, +2.00] | 4/2 | 0.6875 | 1.0000 | no | 10.00 | 208.2 | 0.885 | 0.120 | 0.00 |
| target · K=20 · none · forced | 150 | 73.33 | 3.61 | [65.74, 79.76] | +0.67 | [-3.33, +5.33] | 5/6 | 1.0000 | 1.0000 | no | 20.00 | 216.8 | 0.927 | 0.104 | 0.00 |
| target · K=auto · suppress · natural | 150 | 73.33 | 3.61 | [65.74, 79.76] | +0.67 | [-2.67, +4.00] | 3/4 | 1.0000 | 1.0000 | no | 0.00 | — | — | — | 0.00 |
| target · K=4 · zero · forced | 150 | 72.00 | 3.67 | [64.33, 78.57] | -0.67 | [-2.00, +0.00] | 1/0 | 1.0000 | 1.0000 | yes | 4.00 | 194.7 | 0.919 | 0.148 | 0.00 |
| target · K=4 · noise · forced | 150 | 74.00 | 3.58 | [66.44, 80.36] | +1.33 | [+0.00, +3.33] | 0/2 | 0.5000 | 1.0000 | no | 4.00 | 206.1 | 0.404 | 0.115 | 0.00 |
| target · K=4 · donor · forced | 150 | 72.67 | 3.64 | [65.04, 79.17] | +0.00 | [-2.00, +2.00] | 1/1 | 1.0000 | 1.0000 | yes | 4.00 | 173.2 | 0.855 | 0.155 | 0.00 |
| target · K=4 · pause · forced | 150 | 72.00 | 3.67 | [64.33, 78.57] | -0.67 | [-3.33, +2.00] | 3/2 | 1.0000 | 1.0000 | no | 4.00 | 171.9 | 0.764 | 0.137 | 0.00 |
| target · K=4 · mean · forced | 150 | 72.00 | 3.67 | [64.33, 78.57] | -0.67 | [-3.33, +1.33] | 2/1 | 1.0000 | 1.0000 | yes | 4.00 | 173.5 | 0.859 | 0.156 | 0.00 |

Confirmatory family (7 conditions; Holm within this family):

| Condition | n | Δ vs ref (pts) | 95% bootstrap CI | 90% bootstrap CI | p | p (Holm, family) | Equiv. |
|---|---:|---:|---|---|---:|---:|:---:|
| target · K=0 · none · forced | 150 | -2.67 | [-6.00, +0.00] | [-5.33, +0.00] | 0.2188 | 1.0000 | no |
| target · K=auto · suppress · natural | 150 | +0.67 | [-2.67, +4.00] | [-2.00, +3.33] | 1.0000 | 1.0000 | no |
| target · K=4 · zero · forced | 150 | -0.67 | [-2.00, +0.00] | [-2.00, +0.00] | 1.0000 | 1.0000 | yes |
| target · K=4 · noise · forced | 150 | +1.33 | [+0.00, +3.33] | [+0.00, +3.33] | 0.5000 | 1.0000 | no |
| target · K=4 · donor · forced | 150 | +0.00 | [-2.00, +2.00] | [-1.33, +1.33] | 1.0000 | 1.0000 | yes |
| target · K=4 · pause · forced | 150 | -0.67 | [-3.33, +2.00] | [-3.33, +2.00] | 1.0000 | 1.0000 | no |
| target · K=4 · mean · forced | 150 | -0.67 | [-3.33, +1.33] | [-2.67, +1.33] | 1.0000 | 1.0000 | yes |

Stop-gate telemetry (H2), first latent segment: forced K>4 runs report the share of questions whose model would emit <|latent_end|> right after the 4th latent and the mean step of its first such prediction; auto runs report the share leaving latent mode by a predicted <|latent_end|> and the latent-count histogram:

| Condition | End pred. after 4 (%) | First end-pred. step | Exit by <\|latent_end\|> (%) | Latent counts |
|---|---:|---:|---:|---|
| target · K=5 · none · forced | 2.00 | 4.00 | — | — |
| target · K=10 · none · forced | 2.00 | 4.00 | — | — |
| target · K=20 · none · forced | 2.00 | 4.00 | — | — |

Latent–image alignment (H3): per-question mean over latent steps of max cos(latent, own image tokens) minus max cos(latent, image tokens of a different question); 95% bootstrap CI:

| Condition | n | own − other | 95% CI |
|---|---:|---:|---|
| target · K=1 · none · forced | 149 | 0.0010 | [-0.0029, 0.0050] |
| target · K=3 · none · forced | 149 | 0.0006 | [-0.0032, 0.0047] |
| target · K=4 · none · forced | 149 | 0.0005 | [-0.0032, 0.0044] |
| target · K=5 · none · forced | 149 | 0.0005 | [-0.0030, 0.0042] |
| target · K=10 · none · forced | 149 | 0.0006 | [-0.0020, 0.0034] |
| target · K=20 · none · forced | 149 | 0.0003 | [-0.0020, 0.0026] |
| target · K=4 · zero · forced | 149 | 0.0008 | [-0.0026, 0.0044] |
| target · K=4 · noise · forced | 149 | 0.0009 | [-0.0015, 0.0034] |
| target · K=4 · donor · forced | 149 | 0.0001 | [-0.0035, 0.0040] |
| target · K=4 · pause · forced | 149 | 0.0008 | [-0.0024, 0.0043] |
| target · K=4 · mean · forced | 149 | 0.0002 | [-0.0035, 0.0041] |

Donor runs (H1): among questions whose donor's natural answer differs from their own natural answer, the share answering with the donor's answer (adoption) or their own (retention):

| Condition | n | Adoption (%) | Retention (%) |
|---|---:|---:|---:|
| target · K=4 · donor · forced | 94 | 1.06 | 98.94 |

## mmvp

| Condition | n | Acc (%) | SE | 95% Wilson | Δ vs ref (pts) | 95% bootstrap CI | McNemar b/c | p | p (Holm) | Equiv. | Latent steps | ‖h‖ | cos(h_t,h_t-1) | max cos(h,img) | Unparsed |
|---|---:|---:|---:|---|---:|---|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| base · K=0 · none · natural | 300 | 75.67 | 2.48 | [70.51, 80.18] | +0.00 | [-2.67, +2.67] | 8/8 | 1.0000 | 1.0000 | yes | 0.00 | — | — | — | 0.67 |
| target · K=auto · none · natural | 300 | 75.67 | 2.48 | [70.51, 80.18] | +0.00 | [-1.33, +1.33] | 2/2 | 1.0000 | 1.0000 | yes | 0.00 | — | — | — | 0.00 |
| target · K=0 · none · forced | 300 | 76.00 | 2.47 | [70.86, 80.48] | +0.33 | [-1.00, +1.67] | 2/3 | 1.0000 | 1.0000 | yes | 0.00 | — | — | — | 0.00 |
| target · K=1 · none · forced | 300 | 75.67 | 2.48 | [70.51, 80.18] | +0.00 | [-1.33, +1.33] | 2/2 | 1.0000 | 1.0000 | yes | 1.00 | 154.6 | — | 0.157 | 0.00 |
| target · K=3 · none · forced | 300 | 75.67 | 2.48 | [70.51, 80.18] | +0.00 | [-1.33, +1.33] | 2/2 | 1.0000 | 1.0000 | yes | 3.00 | 153.3 | 0.810 | 0.156 | 0.00 |
| target · K=4 · none · forced **(ref)** | 300 | 75.67 | 2.48 | [70.51, 80.18] | — | — | — | — | — | — | 4.00 | 155.5 | 0.838 | 0.149 | 0.00 |
| target · K=5 · none · forced | 300 | 75.33 | 2.49 | [70.16, 79.87] | -0.33 | [-1.67, +1.00] | 3/2 | 1.0000 | 1.0000 | yes | 5.00 | 162.0 | 0.855 | 0.141 | 0.00 |
| target · K=10 · none · forced | 300 | 75.67 | 2.48 | [70.51, 80.18] | +0.00 | [-1.00, +1.00] | 1/1 | 1.0000 | 1.0000 | yes | 10.00 | 190.1 | 0.883 | 0.113 | 0.00 |
| target · K=20 · none · forced | 300 | 75.00 | 2.50 | [69.80, 79.56] | -0.67 | [-2.33, +1.00] | 4/2 | 0.6875 | 1.0000 | yes | 20.00 | 212.8 | 0.910 | 0.093 | 0.00 |
| target · K=auto · suppress · natural | 300 | 75.67 | 2.48 | [70.51, 80.18] | +0.00 | [-1.33, +1.33] | 2/2 | 1.0000 | 1.0000 | yes | 0.00 | — | — | — | 0.00 |
| target · K=4 · zero · forced | 300 | 74.67 | 2.51 | [69.45, 79.26] | -1.00 | [-3.00, +1.00] | 6/3 | 0.5078 | 1.0000 | yes | 4.00 | 193.3 | 0.930 | 0.141 | 1.67 |
| target · K=4 · noise · forced | 300 | 75.00 | 2.50 | [69.80, 79.56] | -0.67 | [-2.33, +1.00] | 4/2 | 0.6875 | 1.0000 | yes | 4.00 | 202.2 | 0.397 | 0.109 | 0.00 |
| target · K=4 · donor · forced | 300 | 75.67 | 2.48 | [70.51, 80.18] | +0.00 | [-1.00, +1.00] | 1/1 | 1.0000 | 1.0000 | yes | 4.00 | 155.1 | 0.837 | 0.149 | 0.00 |
| target · K=4 · pause · forced | 300 | 75.33 | 2.49 | [70.16, 79.87] | -0.33 | [-1.67, +0.67] | 2/1 | 1.0000 | 1.0000 | yes | 4.00 | 163.3 | 0.801 | 0.128 | 0.00 |
| target · K=4 · mean · forced | 300 | 75.67 | 2.48 | [70.51, 80.18] | +0.00 | [-1.00, +1.00] | 1/1 | 1.0000 | 1.0000 | yes | 4.00 | 153.3 | 0.844 | 0.150 | 0.00 |

Confirmatory family (7 conditions; Holm within this family):

| Condition | n | Δ vs ref (pts) | 95% bootstrap CI | 90% bootstrap CI | p | p (Holm, family) | Equiv. |
|---|---:|---:|---|---|---:|---:|:---:|
| target · K=0 · none · forced | 300 | +0.33 | [-1.00, +1.67] | [-1.00, +1.67] | 1.0000 | 1.0000 | yes |
| target · K=auto · suppress · natural | 300 | +0.00 | [-1.33, +1.33] | [-1.00, +1.00] | 1.0000 | 1.0000 | yes |
| target · K=4 · zero · forced | 300 | -1.00 | [-3.00, +1.00] | [-2.67, +0.67] | 0.5078 | 1.0000 | yes |
| target · K=4 · noise · forced | 300 | -0.67 | [-2.33, +1.00] | [-2.00, +0.67] | 0.6875 | 1.0000 | yes |
| target · K=4 · donor · forced | 300 | +0.00 | [-1.00, +1.00] | [-0.67, +0.67] | 1.0000 | 1.0000 | yes |
| target · K=4 · pause · forced | 300 | -0.33 | [-1.67, +0.67] | [-1.33, +0.67] | 1.0000 | 1.0000 | yes |
| target · K=4 · mean · forced | 300 | +0.00 | [-1.00, +1.00] | [-0.67, +0.67] | 1.0000 | 1.0000 | yes |

Stop-gate telemetry (H2), first latent segment: forced K>4 runs report the share of questions whose model would emit <|latent_end|> right after the 4th latent and the mean step of its first such prediction; auto runs report the share leaving latent mode by a predicted <|latent_end|> and the latent-count histogram:

| Condition | End pred. after 4 (%) | First end-pred. step | Exit by <\|latent_end\|> (%) | Latent counts |
|---|---:|---:|---:|---|
| target · K=5 · none · forced | 0.00 | — | — | — |
| target · K=10 · none · forced | 0.00 | — | — | — |
| target · K=20 · none · forced | 0.00 | — | — | — |

MMVP pair accuracy (both questions of a pair correct):

| Condition | pairs | Pair acc (%) |
|---|---:|---:|
| base · K=0 · none · natural | 150 | 54.67 |
| target · K=auto · none · natural | 150 | 54.00 |
| target · K=0 · none · forced | 150 | 54.67 |
| target · K=1 · none · forced | 150 | 54.00 |
| target · K=3 · none · forced | 150 | 54.00 |
| target · K=4 · none · forced | 150 | 53.33 |
| target · K=5 · none · forced | 150 | 53.33 |
| target · K=10 · none · forced | 150 | 53.33 |
| target · K=20 · none · forced | 150 | 52.00 |
| target · K=auto · suppress · natural | 150 | 54.00 |
| target · K=4 · zero · forced | 150 | 52.67 |
| target · K=4 · noise · forced | 150 | 53.33 |
| target · K=4 · donor · forced | 150 | 53.33 |
| target · K=4 · pause · forced | 150 | 53.33 |
| target · K=4 · mean · forced | 150 | 53.33 |

Latent–image alignment (H3): per-question mean over latent steps of max cos(latent, own image tokens) minus max cos(latent, image tokens of a different question); 95% bootstrap CI:

| Condition | n | own − other | 95% CI |
|---|---:|---:|---|
| target · K=1 · none · forced | 299 | 0.0012 | [-0.0020, 0.0044] |
| target · K=3 · none · forced | 299 | 0.0004 | [-0.0026, 0.0034] |
| target · K=4 · none · forced | 299 | 0.0002 | [-0.0026, 0.0031] |
| target · K=5 · none · forced | 299 | 0.0001 | [-0.0026, 0.0028] |
| target · K=10 · none · forced | 299 | 0.0000 | [-0.0020, 0.0020] |
| target · K=20 · none · forced | 299 | 0.0005 | [-0.0012, 0.0021] |
| target · K=4 · zero · forced | 299 | 0.0010 | [-0.0020, 0.0040] |
| target · K=4 · noise · forced | 299 | 0.0009 | [-0.0008, 0.0026] |
| target · K=4 · donor · forced | 299 | 0.0002 | [-0.0027, 0.0030] |
| target · K=4 · pause · forced | 299 | 0.0016 | [-0.0008, 0.0041] |
| target · K=4 · mean · forced | 299 | 0.0002 | [-0.0027, 0.0030] |

Donor runs (H1): among questions whose donor's natural answer differs from their own natural answer, the share answering with the donor's answer (adoption) or their own (retention):

| Condition | n | Adoption (%) | Retention (%) |
|---|---:|---:|---:|
| target · K=4 · donor · forced | 177 | 1.13 | 98.87 |

## vsp_planning

| Condition | n | Acc (%) | SE | 95% Wilson | Δ vs ref (pts) | 95% bootstrap CI | McNemar b/c | p | p (Holm) | Equiv. | Latent steps | ‖h‖ | cos(h_t,h_t-1) | max cos(h,img) | Unparsed |
|---|---:|---:|---:|---|---:|---|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| base · K=0 · none · natural | 400 | 8.25 | 1.38 | [5.93, 11.36] | -67.25 | [-71.75, -62.75] | 271/2 | 0.0000 | 0.0000 | no | 0.00 | — | — | — | 9.00 |
| target · K=auto · none · natural | 400 | 75.50 | 2.15 | [71.06, 79.46] | +0.00 | [-1.00, +1.00] | 2/2 | 1.0000 | 1.0000 | yes | 3.00 | 57.3 | 0.899 | 0.510 | 0.00 |
| target · K=auto · none · natural · test_py | 400 | 76.25 | 2.13 | [71.84, 80.16] | +0.75 | [-0.50, +2.00] | 2/5 | 0.4531 | 1.0000 | yes | 3.00 | 57.3 | 0.899 | 0.510 | 0.00 |
| target · K=0 · none · forced | 400 | 74.75 | 2.17 | [70.27, 78.76] | -0.75 | [-2.25, +0.75] | 6/3 | 0.5078 | 1.0000 | yes | 0.00 | — | — | — | 0.00 |
| target · K=1 · none · forced | 400 | 75.25 | 2.16 | [70.79, 79.23] | -0.25 | [-1.50, +1.00] | 4/3 | 1.0000 | 1.0000 | yes | 1.00 | 62.4 | — | 0.515 | 0.00 |
| target · K=3 · none · forced | 400 | 75.50 | 2.15 | [71.06, 79.46] | +0.00 | [-0.75, +0.75] | 1/1 | 1.0000 | 1.0000 | yes | 3.00 | 57.2 | 0.899 | 0.510 | 0.00 |
| target · K=4 · none · forced **(ref)** | 400 | 75.50 | 2.15 | [71.06, 79.46] | — | — | — | — | — | — | 4.00 | 95.5 | 0.839 | 0.483 | 0.00 |
| target · K=5 · none · forced | 400 | 74.75 | 2.17 | [70.27, 78.76] | -0.75 | [-2.00, +0.25] | 4/1 | 0.3750 | 1.0000 | yes | 5.00 | 127.7 | 0.845 | 0.449 | 0.00 |
| target · K=10 · none · forced | 400 | 74.50 | 2.18 | [70.01, 78.52] | -1.00 | [-2.50, +0.25] | 6/2 | 0.2891 | 1.0000 | yes | 10.00 | 192.0 | 0.908 | 0.360 | 0.00 |
| target · K=20 · none · forced | 400 | 75.50 | 2.15 | [71.06, 79.46] | +0.00 | [-1.75, +1.75] | 6/6 | 1.0000 | 1.0000 | yes | 20.00 | 217.5 | 0.950 | 0.305 | 0.00 |
| target · K=auto · suppress · natural | 400 | 71.50 | 2.26 | [66.89, 75.70] | -4.00 | [-6.50, -1.50] | 21/5 | 0.0025 | 0.0698 | no | 0.00 | — | — | — | 0.00 |
| target · K=1 · zero · forced | 400 | 75.25 | 2.16 | [70.79, 79.23] | -0.25 | [-1.75, +1.25] | 5/4 | 1.0000 | 1.0000 | yes | 1.00 | 62.4 | — | 0.515 | 0.00 |
| target · K=1 · noise · forced | 400 | 75.00 | 2.17 | [70.53, 78.99] | -0.50 | [-2.00, +0.75] | 5/3 | 0.7266 | 1.0000 | yes | 1.00 | 62.4 | — | 0.515 | 0.00 |
| target · K=1 · donor · forced | 400 | 75.00 | 2.17 | [70.53, 78.99] | -0.50 | [-2.00, +0.75] | 5/3 | 0.7266 | 1.0000 | yes | 1.00 | 62.4 | — | 0.515 | 0.00 |
| target · K=3 · zero · forced | 400 | 75.50 | 2.15 | [71.06, 79.46] | +0.00 | [-1.00, +1.00] | 2/2 | 1.0000 | 1.0000 | yes | 3.00 | 95.0 | 0.877 | 0.479 | 0.00 |
| target · K=3 · noise · forced | 400 | 76.00 | 2.14 | [71.58, 79.93] | +0.50 | [-0.50, +1.50] | 1/3 | 0.6250 | 1.0000 | yes | 3.00 | 101.7 | 0.675 | 0.437 | 0.00 |
| target · K=3 · donor · forced | 400 | 75.50 | 2.15 | [71.06, 79.46] | +0.00 | [-0.75, +0.75] | 1/1 | 1.0000 | 1.0000 | yes | 3.00 | 57.2 | 0.898 | 0.510 | 0.00 |
| target · K=4 · zero · forced | 400 | 75.25 | 2.16 | [70.79, 79.23] | -0.25 | [-1.25, +0.50] | 2/1 | 1.0000 | 1.0000 | yes | 4.00 | 116.5 | 0.903 | 0.457 | 0.00 |
| target · K=4 · noise · forced | 400 | 75.75 | 2.14 | [71.32, 79.69] | +0.25 | [-0.50, +1.25] | 1/2 | 1.0000 | 1.0000 | yes | 4.00 | 125.0 | 0.622 | 0.396 | 0.00 |
| target · K=4 · donor · forced | 400 | 75.25 | 2.16 | [70.79, 79.23] | -0.25 | [-0.75, +0.00] | 1/0 | 1.0000 | 1.0000 | yes | 4.00 | 95.4 | 0.839 | 0.483 | 0.00 |
| target · K=4 · pause · forced | 400 | 75.50 | 2.15 | [71.06, 79.46] | +0.00 | [-0.75, +0.75] | 1/1 | 1.0000 | 1.0000 | yes | 4.00 | 146.1 | 0.807 | 0.429 | 0.00 |
| target · K=4 · mean · forced | 400 | 75.50 | 2.15 | [71.06, 79.46] | +0.00 | [-0.75, +0.75] | 1/1 | 1.0000 | 1.0000 | yes | 4.00 | 95.4 | 0.839 | 0.483 | 0.00 |
| target · K=5 · zero · forced | 400 | 75.00 | 2.17 | [70.53, 78.99] | -0.50 | [-1.25, +0.00] | 2/0 | 0.5000 | 1.0000 | yes | 5.00 | 134.5 | 0.922 | 0.437 | 0.00 |
| target · K=5 · noise · forced | 400 | 75.50 | 2.15 | [71.06, 79.46] | +0.00 | [-1.00, +1.00] | 2/2 | 1.0000 | 1.0000 | yes | 5.00 | 143.5 | 0.571 | 0.362 | 0.00 |
| target · K=5 · donor · forced | 400 | 74.75 | 2.17 | [70.27, 78.76] | -0.75 | [-1.75, +0.00] | 3/0 | 0.2500 | 1.0000 | yes | 5.00 | 127.6 | 0.845 | 0.449 | 0.00 |
| target · K=10 · zero · forced | 400 | 73.50 | 2.21 | [68.97, 77.59] | -2.00 | [-4.00, +0.00] | 12/4 | 0.0768 | 1.0000 | no | 10.03 | 169.6 | 0.959 | 0.391 | 1.50 |
| target · K=10 · noise · forced | 400 | 75.00 | 2.17 | [70.53, 78.99] | -0.50 | [-2.00, +1.00] | 6/4 | 0.7539 | 1.0000 | yes | 10.00 | 180.5 | 0.460 | 0.281 | 0.00 |
| target · K=10 · donor · forced | 400 | 74.50 | 2.18 | [70.01, 78.52] | -1.00 | [-2.50, +0.25] | 6/2 | 0.2891 | 1.0000 | yes | 10.00 | 192.1 | 0.908 | 0.360 | 0.00 |
| target · K=20 · zero · forced | 400 | 20.00 | 2.00 | [16.37, 24.20] | -55.50 | [-60.75, -50.25] | 227/5 | 0.0000 | 0.0000 | no | 21.55 | 170.5 | 0.964 | 0.363 | 65.00 |
| target · K=20 · noise · forced | 400 | 75.25 | 2.16 | [70.79, 79.23] | -0.25 | [-2.00, +1.50] | 7/6 | 1.0000 | 1.0000 | yes | 20.00 | 198.6 | 0.408 | 0.230 | 0.00 |
| target · K=20 · donor · forced | 400 | 76.00 | 2.14 | [71.58, 79.93] | +0.50 | [-1.00, +2.00] | 4/6 | 0.7539 | 1.0000 | yes | 20.00 | 218.2 | 0.950 | 0.305 | 0.00 |

Confirmatory family (7 conditions; Holm within this family):

| Condition | n | Δ vs ref (pts) | 95% bootstrap CI | 90% bootstrap CI | p | p (Holm, family) | Equiv. |
|---|---:|---:|---|---|---:|---:|:---:|
| target · K=0 · none · forced | 400 | -0.75 | [-2.25, +0.75] | [-2.00, +0.50] | 0.5078 | 1.0000 | yes |
| target · K=auto · suppress · natural | 400 | -4.00 | [-6.50, -1.50] | [-6.00, -2.00] | 0.0025 | 0.0175 | no |
| target · K=4 · zero · forced | 400 | -0.25 | [-1.25, +0.50] | [-1.00, +0.50] | 1.0000 | 1.0000 | yes |
| target · K=4 · noise · forced | 400 | +0.25 | [-0.50, +1.25] | [-0.50, +1.00] | 1.0000 | 1.0000 | yes |
| target · K=4 · donor · forced | 400 | -0.25 | [-0.75, +0.00] | [-0.75, +0.00] | 1.0000 | 1.0000 | yes |
| target · K=4 · pause · forced | 400 | +0.00 | [-0.75, +0.75] | [-0.50, +0.50] | 1.0000 | 1.0000 | yes |
| target · K=4 · mean · forced | 400 | +0.00 | [-0.75, +0.75] | [-0.50, +0.50] | 1.0000 | 1.0000 | yes |

Stop-gate telemetry (H2), first latent segment: forced K>4 runs report the share of questions whose model would emit <|latent_end|> right after the 4th latent and the mean step of its first such prediction; auto runs report the share leaving latent mode by a predicted <|latent_end|> and the latent-count histogram:

| Condition | End pred. after 4 (%) | First end-pred. step | Exit by <\|latent_end\|> (%) | Latent counts |
|---|---:|---:|---:|---|
| target · K=auto · none · natural | — | — | 100.00 | 3:400 |
| target · K=auto · none · natural · test_py | — | — | 100.00 | 3:400 |
| target · K=5 · none · forced | 100.00 | 3.00 | — | — |
| target · K=10 · none · forced | 100.00 | 3.00 | — | — |
| target · K=20 · none · forced | 100.00 | 3.00 | — | — |
| target · K=5 · zero · forced | 0.25 | 4.00 | — | — |
| target · K=5 · noise · forced | 1.75 | 2.87 | — | — |
| target · K=5 · donor · forced | 100.00 | 3.00 | — | — |
| target · K=10 · zero · forced | 0.25 | 6.00 | — | — |
| target · K=10 · noise · forced | 1.75 | 3.71 | — | — |
| target · K=10 · donor · forced | 100.00 | 3.00 | — | — |
| target · K=20 · zero · forced | 0.25 | 6.00 | — | — |
| target · K=20 · noise · forced | 1.75 | 4.03 | — | — |
| target · K=20 · donor · forced | 100.00 | 3.00 | — | — |

Per-category accuracy (%):

| Condition | level3 | level4 | level5 | level6 |
|---|---:|---:|---:|---:|
| base · K=0 · none · natural | 19.00 | 9.00 | 2.00 | 3.00 |
| target · K=auto · none · natural | 91.00 | 83.00 | 77.00 | 51.00 |
| target · K=auto · none · natural · test_py | 92.00 | 82.00 | 77.00 | 54.00 |
| target · K=0 · none · forced | 91.00 | 82.00 | 76.00 | 50.00 |
| target · K=1 · none · forced | 91.00 | 83.00 | 76.00 | 51.00 |
| target · K=3 · none · forced | 91.00 | 83.00 | 77.00 | 51.00 |
| target · K=4 · none · forced | 92.00 | 83.00 | 76.00 | 51.00 |
| target · K=5 · none · forced | 91.00 | 82.00 | 76.00 | 50.00 |
| target · K=10 · none · forced | 91.00 | 81.00 | 75.00 | 51.00 |
| target · K=20 · none · forced | 91.00 | 81.00 | 78.00 | 52.00 |
| target · K=auto · suppress · natural | 81.00 | 82.00 | 72.00 | 51.00 |
| target · K=1 · zero · forced | 92.00 | 82.00 | 76.00 | 51.00 |
| target · K=1 · noise · forced | 92.00 | 82.00 | 75.00 | 51.00 |
| target · K=1 · donor · forced | 91.00 | 83.00 | 75.00 | 51.00 |
| target · K=3 · zero · forced | 91.00 | 83.00 | 77.00 | 51.00 |
| target · K=3 · noise · forced | 92.00 | 83.00 | 77.00 | 52.00 |
| target · K=3 · donor · forced | 91.00 | 83.00 | 77.00 | 51.00 |
| target · K=4 · zero · forced | 91.00 | 83.00 | 76.00 | 51.00 |
| target · K=4 · noise · forced | 92.00 | 83.00 | 76.00 | 52.00 |
| target · K=4 · donor · forced | 91.00 | 83.00 | 76.00 | 51.00 |
| target · K=4 · pause · forced | 91.00 | 83.00 | 76.00 | 52.00 |
| target · K=4 · mean · forced | 92.00 | 83.00 | 76.00 | 51.00 |
| target · K=5 · zero · forced | 92.00 | 82.00 | 76.00 | 50.00 |
| target · K=5 · noise · forced | 92.00 | 83.00 | 76.00 | 51.00 |
| target · K=5 · donor · forced | 92.00 | 83.00 | 75.00 | 49.00 |
| target · K=10 · zero · forced | 84.00 | 82.00 | 76.00 | 52.00 |
| target · K=10 · noise · forced | 91.00 | 82.00 | 77.00 | 50.00 |
| target · K=10 · donor · forced | 91.00 | 81.00 | 75.00 | 51.00 |
| target · K=20 · zero · forced | 7.00 | 6.00 | 15.00 | 52.00 |
| target · K=20 · noise · forced | 91.00 | 83.00 | 76.00 | 51.00 |
| target · K=20 · donor · forced | 92.00 | 82.00 | 78.00 | 52.00 |

Latent–image alignment (H3): per-question mean over latent steps of max cos(latent, own image tokens) minus max cos(latent, image tokens of a different question); 95% bootstrap CI:

| Condition | n | own − other | 95% CI |
|---|---:|---:|---|
| target · K=auto · none · natural | 399 | 0.0007 | [-0.0012, 0.0025] |
| target · K=auto · none · natural · test_py | 399 | 0.0007 | [-0.0012, 0.0025] |
| target · K=1 · none · forced | 399 | 0.0009 | [-0.0013, 0.0032] |
| target · K=3 · none · forced | 399 | 0.0007 | [-0.0012, 0.0025] |
| target · K=4 · none · forced | 399 | 0.0005 | [-0.0012, 0.0023] |
| target · K=5 · none · forced | 399 | 0.0005 | [-0.0012, 0.0021] |
| target · K=10 · none · forced | 399 | 0.0004 | [-0.0012, 0.0019] |
| target · K=20 · none · forced | 399 | 0.0002 | [-0.0013, 0.0018] |
| target · K=1 · zero · forced | 399 | 0.0009 | [-0.0013, 0.0032] |
| target · K=1 · noise · forced | 399 | 0.0009 | [-0.0013, 0.0032] |
| target · K=1 · donor · forced | 399 | 0.0009 | [-0.0013, 0.0032] |
| target · K=3 · zero · forced | 399 | 0.0005 | [-0.0015, 0.0025] |
| target · K=3 · noise · forced | 399 | 0.0006 | [-0.0010, 0.0022] |
| target · K=3 · donor · forced | 399 | 0.0006 | [-0.0013, 0.0024] |
| target · K=4 · zero · forced | 399 | 0.0005 | [-0.0014, 0.0024] |
| target · K=4 · noise · forced | 399 | 0.0006 | [-0.0009, 0.0020] |
| target · K=4 · donor · forced | 399 | 0.0005 | [-0.0012, 0.0022] |
| target · K=4 · pause · forced | 399 | 0.0005 | [-0.0011, 0.0022] |
| target · K=4 · mean · forced | 399 | 0.0005 | [-0.0012, 0.0023] |
| target · K=5 · zero · forced | 399 | 0.0004 | [-0.0014, 0.0023] |
| target · K=5 · noise · forced | 399 | 0.0005 | [-0.0007, 0.0018] |
| target · K=5 · donor · forced | 399 | 0.0004 | [-0.0012, 0.0021] |
| target · K=10 · zero · forced | 399 | 0.0003 | [-0.0014, 0.0020] |
| target · K=10 · noise · forced | 399 | 0.0002 | [-0.0007, 0.0012] |
| target · K=10 · donor · forced | 399 | 0.0003 | [-0.0012, 0.0019] |
| target · K=20 · zero · forced | 399 | 0.0002 | [-0.0015, 0.0019] |
| target · K=20 · noise · forced | 399 | -0.0001 | [-0.0008, 0.0007] |
| target · K=20 · donor · forced | 399 | 0.0001 | [-0.0014, 0.0016] |

Donor runs (H1): among questions whose donor's natural answer differs from their own natural answer, the share answering with the donor's answer (adoption) or their own (retention):

| Condition | n | Adoption (%) | Retention (%) |
|---|---:|---:|---:|
| target · K=1 · donor · forced | 393 | 0.00 | 98.73 |
| target · K=3 · donor · forced | 393 | 0.00 | 99.49 |
| target · K=4 · donor · forced | 393 | 0.00 | 98.22 |
| target · K=5 · donor · forced | 393 | 0.00 | 97.71 |
| target · K=10 · donor · forced | 392 | 0.00 | 98.98 |
| target · K=20 · donor · forced | 393 | 0.00 | 98.47 |

## vstar

| Condition | n | Acc (%) | SE | 95% Wilson | Δ vs ref (pts) | 95% bootstrap CI | McNemar b/c | p | p (Holm) | Equiv. | Latent steps | ‖h‖ | cos(h_t,h_t-1) | max cos(h,img) | Unparsed |
|---|---:|---:|---:|---|---:|---|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| base · K=0 · none · natural | 191 | 65.45 | 3.44 | [58.46, 71.82] | -3.66 | [-8.90, +1.57] | 16/9 | 0.2295 | 1.0000 | no | 0.00 | — | — | — | 10.99 |
| target · K=auto · none · natural | 191 | 67.02 | 3.40 | [60.07, 73.29] | -2.09 | [-4.19, -0.52] | 4/0 | 0.1250 | 1.0000 | no | 0.00 | — | — | — | 1.57 |
| target · K=0 · none · forced | 191 | 68.59 | 3.36 | [61.69, 74.75] | -0.52 | [-1.57, +0.00] | 1/0 | 1.0000 | 1.0000 | yes | 0.00 | — | — | — | 0.00 |
| target · K=1 · none · forced | 191 | 70.16 | 3.31 | [63.32, 76.20] | +1.05 | [+0.00, +2.62] | 0/2 | 0.5000 | 1.0000 | yes | 1.00 | 180.1 | — | 0.174 | 0.00 |
| target · K=3 · none · forced | 191 | 69.11 | 3.34 | [62.23, 75.23] | +0.00 | [+0.00, +0.00] | 0/0 | 1.0000 | 1.0000 | yes | 3.00 | 165.5 | 0.804 | 0.174 | 0.00 |
| target · K=4 · none · forced **(ref)** | 191 | 69.11 | 3.34 | [62.23, 75.23] | — | — | — | — | — | — | 4.00 | 165.2 | 0.829 | 0.169 | 0.00 |
| target · K=5 · none · forced | 191 | 69.63 | 3.33 | [62.78, 75.71] | +0.52 | [+0.00, +1.57] | 0/1 | 1.0000 | 1.0000 | yes | 5.00 | 169.3 | 0.842 | 0.163 | 0.00 |
| target · K=10 · none · forced | 191 | 69.63 | 3.33 | [62.78, 75.71] | +0.52 | [-1.57, +2.62] | 2/3 | 1.0000 | 1.0000 | yes | 10.00 | 181.3 | 0.879 | 0.135 | 0.00 |
| target · K=20 · none · forced | 191 | 67.54 | 3.39 | [60.61, 73.78] | -1.57 | [-3.66, +0.00] | 3/0 | 0.2500 | 1.0000 | no | 20.00 | 206.7 | 0.917 | 0.113 | 1.05 |
| target · K=auto · suppress · natural | 191 | 67.02 | 3.40 | [60.07, 73.29] | -2.09 | [-4.19, -0.52] | 4/0 | 0.1250 | 1.0000 | no | 0.00 | — | — | — | 1.57 |
| target · K=4 · zero · forced | 191 | 42.41 | 3.58 | [35.62, 49.50] | -26.70 | [-33.51, -19.90] | 53/2 | 0.0000 | 0.0000 | no | 4.00 | 203.8 | 0.916 | 0.156 | 40.84 |
| target · K=4 · noise · forced | 191 | 68.59 | 3.36 | [61.69, 74.75] | -0.52 | [-1.57, +0.00] | 1/0 | 1.0000 | 1.0000 | yes | 4.00 | 209.1 | 0.400 | 0.130 | 0.00 |
| target · K=4 · donor · forced | 191 | 69.11 | 3.34 | [62.23, 75.23] | +0.00 | [-1.57, +1.57] | 1/1 | 1.0000 | 1.0000 | yes | 4.00 | 164.9 | 0.829 | 0.169 | 0.00 |
| target · K=4 · pause · forced | 191 | 69.11 | 3.34 | [62.23, 75.23] | +0.00 | [+0.00, +0.00] | 0/0 | 1.0000 | 1.0000 | yes | 4.00 | 170.0 | 0.774 | 0.156 | 0.00 |
| target · K=4 · mean · forced | 191 | 68.59 | 3.36 | [61.69, 74.75] | -0.52 | [-1.57, +0.00] | 1/0 | 1.0000 | 1.0000 | yes | 4.00 | 163.7 | 0.837 | 0.171 | 0.00 |

Confirmatory family (7 conditions; Holm within this family):

| Condition | n | Δ vs ref (pts) | 95% bootstrap CI | 90% bootstrap CI | p | p (Holm, family) | Equiv. |
|---|---:|---:|---|---|---:|---:|:---:|
| target · K=0 · none · forced | 191 | -0.52 | [-1.57, +0.00] | [-1.57, +0.00] | 1.0000 | 1.0000 | yes |
| target · K=auto · suppress · natural | 191 | -2.09 | [-4.19, -0.52] | [-4.19, -0.52] | 0.1250 | 0.7500 | no |
| target · K=4 · zero · forced | 191 | -26.70 | [-33.51, -19.90] | [-32.46, -20.94] | 0.0000 | 0.0000 | no |
| target · K=4 · noise · forced | 191 | -0.52 | [-1.57, +0.00] | [-1.57, +0.00] | 1.0000 | 1.0000 | yes |
| target · K=4 · donor · forced | 191 | +0.00 | [-1.57, +1.57] | [-1.05, +1.05] | 1.0000 | 1.0000 | yes |
| target · K=4 · pause · forced | 191 | +0.00 | [+0.00, +0.00] | [+0.00, +0.00] | 1.0000 | 1.0000 | yes |
| target · K=4 · mean · forced | 191 | -0.52 | [-1.57, +0.00] | [-1.57, +0.00] | 1.0000 | 1.0000 | yes |

Stop-gate telemetry (H2), first latent segment: forced K>4 runs report the share of questions whose model would emit <|latent_end|> right after the 4th latent and the mean step of its first such prediction; auto runs report the share leaving latent mode by a predicted <|latent_end|> and the latent-count histogram:

| Condition | End pred. after 4 (%) | First end-pred. step | Exit by <\|latent_end\|> (%) | Latent counts |
|---|---:|---:|---:|---|
| target · K=5 · none · forced | 0.00 | — | — | — |
| target · K=10 · none · forced | 0.00 | 5.00 | — | — |
| target · K=20 · none · forced | 0.00 | 5.00 | — | — |

Per-category accuracy (%):

| Condition | direct_attributes | relative_position |
|---|---:|---:|
| base · K=0 · none · natural | 66.09 | 64.47 |
| target · K=auto · none · natural | 70.43 | 61.84 |
| target · K=0 · none · forced | 71.30 | 64.47 |
| target · K=1 · none · forced | 71.30 | 68.42 |
| target · K=3 · none · forced | 71.30 | 65.79 |
| target · K=4 · none · forced | 71.30 | 65.79 |
| target · K=5 · none · forced | 72.17 | 65.79 |
| target · K=10 · none · forced | 71.30 | 67.11 |
| target · K=20 · none · forced | 69.57 | 64.47 |
| target · K=auto · suppress · natural | 70.43 | 61.84 |
| target · K=4 · zero · forced | 35.65 | 52.63 |
| target · K=4 · noise · forced | 71.30 | 64.47 |
| target · K=4 · donor · forced | 71.30 | 65.79 |
| target · K=4 · pause · forced | 71.30 | 65.79 |
| target · K=4 · mean · forced | 70.43 | 65.79 |

Latent–image alignment (H3): per-question mean over latent steps of max cos(latent, own image tokens) minus max cos(latent, image tokens of a different question); 95% bootstrap CI:

| Condition | n | own − other | 95% CI |
|---|---:|---:|---|
| target · K=1 · none · forced | 190 | 0.0011 | [-0.0019, 0.0040] |
| target · K=3 · none · forced | 190 | 0.0004 | [-0.0029, 0.0037] |
| target · K=4 · none · forced | 190 | 0.0003 | [-0.0028, 0.0035] |
| target · K=5 · none · forced | 190 | 0.0006 | [-0.0024, 0.0036] |
| target · K=10 · none · forced | 190 | -0.0003 | [-0.0027, 0.0022] |
| target · K=20 · none · forced | 190 | -0.0004 | [-0.0024, 0.0016] |
| target · K=4 · zero · forced | 190 | 0.0005 | [-0.0026, 0.0037] |
| target · K=4 · noise · forced | 190 | 0.0008 | [-0.0012, 0.0029] |
| target · K=4 · donor · forced | 190 | 0.0006 | [-0.0025, 0.0038] |
| target · K=4 · pause · forced | 190 | 0.0006 | [-0.0022, 0.0034] |
| target · K=4 · mean · forced | 190 | 0.0002 | [-0.0030, 0.0035] |

Donor runs (H1): among questions whose donor's natural answer differs from their own natural answer, the share answering with the donor's answer (adoption) or their own (retention):

| Condition | n | Adoption (%) | Retention (%) |
|---|---:|---:|---:|
| target · K=4 · donor · forced | 157 | 0.00 | 99.36 |

## Notes

- Δ, b/c and p use only questions present in both the condition and the reference (paired).
- b = reference correct & condition wrong; c = reference wrong & condition correct.
- ‖h‖, cos(h_t,h_t-1), max cos(h,img): means over latent steps of the natural latent (the hidden state the model would feed), its cosine to the previous step, and its maximum cosine to any image-token embedding of the prompt.
- `Equiv.` = yes when the 90% paired-bootstrap CI of Δ falls within the equivalence margin (two one-sided tests).
