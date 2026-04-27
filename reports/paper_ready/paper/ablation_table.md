| Run | Model | Train Datasets | Eval Datasets | State Accuracy | State Macro F1 | Future Hesitation AUPRC | Future Correction AUPRC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Feature subset ablation | rules | chico,ha_vid | chico,ha_vid | 0.1968 | 0.1251 | 0.8663 | 0.6365 |
| Harmonization sensitivity ablation | rules | chico,ha_vid | chico,ha_vid | 0.2000 | 0.1262 | 0.8663 | 0.6365 |
| Short-horizon label ablation | rules | chico,ha_vid | chico,ha_vid | 0.1966 | 0.1246 | 0.7461 | 0.4460 |
| Feature subset ablation | classical | chico,ha_vid | chico,ha_vid | 0.2768 | 0.2260 | 0.9125 | 0.6151 |
| Harmonization sensitivity ablation | classical | chico,ha_vid | chico,ha_vid | 0.2876 | 0.2444 | 0.9244 | 0.6181 |
| Short-horizon label ablation | classical | chico,ha_vid | chico,ha_vid | 0.2825 | 0.2392 | 0.7837 | 0.4737 |
| Feature subset ablation | deep | chico,ha_vid | chico,ha_vid | 0.4605 | 0.2848 | 0.9453 | 0.6791 |
| Harmonization sensitivity ablation | deep | chico,ha_vid | chico,ha_vid | 0.4314 | 0.2537 | 0.9076 | 0.6360 |
| Short-horizon label ablation | deep | chico,ha_vid | chico,ha_vid | 0.3061 | 0.1885 | 0.7938 | 0.4757 |