
import json
import numpy as np
import re

# Parameters
D0 = 5000      # distance threshold in meters
LAMBDA = 0.0002  # decay rate

def spatial_score(d):
    if d <= D0:
        return 1 - 0.5 * (d / D0)
    else:
        return 0.5 * np.exp(-LAMBDA * (d - D0))

def extract_distance(transit_info):
    # Extracts the first number followed by 'm' or 'm away'
    match = re.search(r'(\d+(?:\.\d+)?)\s*m', transit_info)
    if match:
        return float(match.group(1))
    return None

def calculate_spatial_score(plan_dict):
    plan = plan_dict.get("plan", [])
    total_day_scores = []
    
    for day in plan:
        poi_scores = []
        poi_list = day.get("point_of_interest_list", "")
        for segment in poi_list.split(";"):
            if "nearest transit:" in segment:
                transit_info = segment.split("nearest transit:")[1]
                distance = extract_distance(transit_info)
                if distance is not None:
                    poi_scores.append(spatial_score(distance))

        if poi_scores:
            avg_day_score = sum(poi_scores) / len(poi_scores)
            total_day_scores.append(avg_day_score)

    if total_day_scores:
        return sum(total_day_scores) / len(total_day_scores)
    else:
        return 0.0

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

def compute_average_spatial_score(file_path):
    data = load_jsonl(file_path)
    total = 0.0
    count = 0

    for entry in data:
        try:
            # handle nested plans
            if "phi4_direct_og_sole-planning_results" in entry:
                entry = entry["phi4_direct_og_sole-planning_results"]
            score = calculate_spatial_score(entry)
            total += score
            count += 1
        except Exception as e:
            print(f"Skipping due to error: {e}")

    return total / count if count > 0 else 0.0

def main():
    annotation_file = "annotation_plan_5day.jsonl" #add path
    revised_file = "3day_phi4_eval_structured.jsonl"

    ann_score = compute_average_spatial_score(annotation_file)
    rev_score = compute_average_spatial_score(revised_file)
    delta_score = ann_score - rev_score

    print(f"Mean Annotation Spatial Score: {ann_score:.4f}")
    print(f"Mean Revised Spatial Score:    {rev_score:.4f}")
    print(f"Spatial Adaptability (Aspa):   {delta_score:.4f}")

if __name__ == "__main__":
    main()
