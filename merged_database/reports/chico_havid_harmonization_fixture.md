# CHICO vs HA-ViD harmonization report

## Field coverage gaps
| Field | CHICO | HA-ViD | Gap |
|---|---:|---:|---:|
| timestamp_ms | 1.00 | 1.00 | 0.00 |
| hand_left | 1.00 | 1.00 | 0.00 |
| hand_right | 0.00 | 1.00 | 1.00 |
| pose_confidence | 1.00 | 1.00 | 0.00 |
| task_step | 1.00 | 1.00 | 0.00 |
| action_label_raw | 1.00 | 1.00 | 0.00 |
| canonical_action_label | 1.00 | 1.00 | 0.00 |
| human_robot_distance | 1.00 | 1.00 | 0.00 |
| tool | 0.00 | 0.69 | 0.69 |
| manipulated_object | 0.00 | 1.00 | 1.00 |

## Transfer issues

- coverage_gap:hand_right:1.00
- coverage_gap:tool:0.69
- coverage_gap:manipulated_object:1.00