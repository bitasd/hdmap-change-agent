# HD Map Change Detection

Does the HD map still match the road? This project compares vehicle LiDAR against
[Argoverse 2](https://argoverse.github.io/user-guide/datasets/map_change_detection.html) HD maps
to measure how far lane lines and crosswalks are from where the map says, with a confidence score
for each, and (later) wraps it in an agent you can ask questions in plain English.

![Detected lane paint vs the HD map](docs/figures/07_paint_on_map.png)

**Step-by-step walkthrough with figures: [docs/walkthrough.md](docs/walkthrough.md)**

Status and plan: [ROADMAP.md](ROADMAP.md). So far: LiDAR in the map frame, ground segmentation,
lane-paint extraction. Next: per-element deviation in cm.
