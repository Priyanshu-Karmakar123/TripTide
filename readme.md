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
pip install -r requirements.txt

# generate plans (example)
python tools/planner/run.py --model qwen --days 5 \
  --input data/inputs_5day.jsonl --out runs/qwen_5day.jsonl

# evaluate
python eval/eval.py --pred runs/qwen_5day.jsonl \
  --gold data/annotations/annotation_plan_5day.jsonl
