# Phase 3 judge meta-evaluation (v1, n=60)

sample strata: unstratified (60 rows)

## 1. Judge vs hand-labeled gold (default arm, gpt-4o-mini answers)

| judge | criterion | kappa | 95% CI | acc | sens | spec | gold_rate |
|---|---|---|---|---|---|---|---|
| gemini-2.5-flash | faithfulness | undefined | (constant gold) | 0.833 | - | - | 1.000 |
| gemini-2.5-flash | answer_correctness | 0.960 | [+0.86,+1.00] | 0.983 | 1.000 | 0.977 | 0.283 |
| gpt-4o-mini | faithfulness | undefined | (constant gold) | 0.283 | - | - | 1.000 |
| gpt-4o-mini | answer_correctness | 1.000 | [+1.00,+1.00] | 1.000 | 1.000 | 1.000 | 0.283 |
| claude-haiku-4.5 | faithfulness | undefined | (constant gold) | 0.633 | - | - | 1.000 |
| claude-haiku-4.5 | answer_correctness | 1.000 | [+1.00,+1.00] | 1.000 | 1.000 | 1.000 | 0.283 |
| qwen-2.5-72b | faithfulness | undefined | (constant gold) | 0.667 | - | - | 1.000 |
| qwen-2.5-72b | answer_correctness | 0.812 | [+0.64,+0.96] | 0.917 | 1.000 | 0.884 | 0.283 |

## 2. Judge x judge agreement (default arm)

Each cell is `kappa/accuracy` on the binary labels: faithfulness, then answer_correctness.

| judge | gemini-2.5-flash | gpt-4o-mini | claude-haiku-4.5 | qwen-2.5-72b |
|---|---|---|---|---|
| gemini-2.5-flash | 1.000/1.000 | 0.147/0.960 | 0.270/0.960 | 0.143/0.776 |
| gpt-4o-mini | 0.147/0.960 | 1.000/1.000 | 0.313/1.000 | 0.330/0.812 |
| claude-haiku-4.5 | 0.270/0.960 | 0.313/1.000 | 1.000/1.000 | 0.341/0.812 |
| qwen-2.5-72b | 0.143/0.776 | 0.330/0.812 | 0.341/0.812 | 1.000/1.000 |

## 3. Content-fixed self-preference (same gpt answer, claimed author varies)

| judge | criterion | gpt label | claude label | delta |
|---|---|---|---|---|
| gemini-2.5-flash | faithfulness | 0.933 | 0.867 | +0.067 |
| gemini-2.5-flash | answer_correctness | 0.300 | 0.300 | +0.000 |
| gpt-4o-mini | faithfulness | 0.283 | 0.283 | +0.000 |
| gpt-4o-mini | answer_correctness | 0.283 | 0.283 | +0.000 |
| claude-haiku-4.5 | faithfulness | 0.674 | 0.756 | -0.082 |
| claude-haiku-4.5 | answer_correctness | 0.283 | 0.283 | +0.000 |
| qwen-2.5-72b | faithfulness | 0.745 | 0.731 | +0.014 |
| qwen-2.5-72b | answer_correctness | 0.355 | 0.377 | -0.022 |

## 4. Answer-set difference (NOT self-preference: different answers)

| judge | criterion | gpt answers | claude answers | delta |
|---|---|---|---|---|
| gemini-2.5-flash | faithfulness | 0.822 | 0.972 | -0.150 |
| gemini-2.5-flash | answer_correctness | 0.300 | 0.317 | -0.017 |
| gpt-4o-mini | faithfulness | 0.283 | 0.352 | -0.068 |
| gpt-4o-mini | answer_correctness | 0.283 | 0.150 | +0.133 |
| claude-haiku-4.5 | faithfulness | 0.538 | 0.923 | -0.385 |
| claude-haiku-4.5 | answer_correctness | 0.283 | 0.143 | +0.141 |
| qwen-2.5-72b | faithfulness | 0.609 | 0.778 | -0.169 |
| qwen-2.5-72b | answer_correctness | 0.330 | 0.327 | +0.003 |

## 5. Prompt framing (gemini, gpt-4o-mini answers)

| criterion | variant | kappa vs default | mean delta |
|---|---|---|---|
| faithfulness | reversed | 0.400 | +0.012 |
| faithfulness | terse | 0.147 | -0.555 |
| answer_correctness | reversed | 1.000 | +0.000 |
| answer_correctness | terse | 0.231 | +0.467 |

## 6. Test-retest (gemini, gpt-4o-mini answers)

| temperature | criterion | runs | alpha | flipped rows |
|---|---|---|---|---|
| 0.0 | faithfulness | 2 | 1.000 | 0/60 |
| 0.0 | answer_correctness | 2 | 1.000 | 0/60 |
| 0.7 | faithfulness | 3 | 0.518 | 11/60 |
| 0.7 | answer_correctness | 3 | 0.898 | 4/60 |

## 7. Cross-run drift (original v1 run vs this re-run, same prompt, temperature 0)

| judge | criterion | changed | mean delta |
|---|---|---|---|
| gemini-2.5-flash | faithfulness | 1/60 | +0.017 |
| gemini-2.5-flash | answer_correctness | 1/60 | +0.017 |
| gpt-4o-mini | faithfulness | 0/60 | +0.000 |
| gpt-4o-mini | answer_correctness | 0/60 | +0.000 |
| claude-haiku-4.5 | faithfulness | 1/60 | -0.003 |
| claude-haiku-4.5 | answer_correctness | 0/60 | +0.000 |
| qwen-2.5-72b | faithfulness | 1/60 | +0.006 |
| qwen-2.5-72b | answer_correctness | 2/60 | +0.005 |
