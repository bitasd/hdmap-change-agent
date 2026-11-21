# Roadmap — HD Map Change Detection

Each phase ends with something you can show (plot, number, demo).

| # | Phase | What we do | Deliverable | Status |
|---|---|---|---|---|
| 0 | Setup | Repo, env, download one TbV log (no cameras) | Working env + data | ✅ Done |
| 1 | See the data | LiDAR in city frame + HD map overlay (PNG), then Rerun 3D viewer | `docs/lidar_on_map.png`, `src/view_rerun.py` | ✅ Done |
| 2 | Find the paint | Separate ground points, keep high-intensity points (lane paint) | `docs/paint_on_map.png`, `src/paint.py` | ✅ Done |
| 3 | Measure deviation | Distance from paint points to each map lane boundary / crosswalk | Per-element error table (cm) | ⬜ |
| 4 | Confidence | Score each element: point coverage, occlusion, intensity noise | Error + confidence per element | ⬜ |
| 5 | Eval | (a) Shift map elements by 10/25/50 cm and check recovery; (b) TbV match/mismatch labels → precision/recall | Eval report + baseline numbers | ⬜ |
| 6 | Post-training: calibration | Make "90% confident" mean right 90% of the time (temperature scaling / isotonic) | Calibration curve before/after | ⬜ |
| 7 | Post-training: fine-tune | Fine-tune a pretrained lane-paint segmentation model where intensity thresholds fail | Baseline vs fine-tuned comparison | ⬜ |
| 8 | Agent | NL question → tools (load log, get element, compute deviation) → short report (LangGraph + LLM) | Chat demo | ⬜ |
| 9 | VLM on cameras (optional) | Look at camera images of flagged changes: why did it change? | "Construction zone, temporary" style labels | ⬜ |
| 10 | Polish | README, GIF from Rerun, results table, write-up / blog post | Public GitHub repo | ⬜ |

## Rules of thumb
- One log until phase 5; then a handful of match + mismatch logs.
- Eval starts in phase 3, not at the end.
- Only fine-tune where phase 5 shows a measurable failure.
- Push to GitHub after every phase.
