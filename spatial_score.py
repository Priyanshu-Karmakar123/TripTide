"""
import json
import numpy as np

def lin_exp_score(distance):
    if distance <= 5000:
        return 1 - 0.5 * (distance / 5000)
    else:
        return 0.5 * np.exp(-0.0002 * (distance - 5000))

def calculate_spatial_score(travel_plan):
    total_days_score = 0
    total_days = 0

    for day in travel_plan["plan"]:
        point_of_interest_list = day.get("point_of_interest_list", "")
        pois = point_of_interest_list.split(";")
        day_scores = []

        for poi in pois:
            if "nearest transit:" in poi:
                transit_info = poi.split("nearest transit:")[1].strip()
                if "," in transit_info:
                    distance_str = transit_info.split(",")[-1].strip().split("m")[0].strip()
                    try:
                        distance = float(distance_str)
                        day_scores.append(lin_exp_score(distance))
                    except ValueError:
                        continue

        if day_scores:
            average_day_score = np.mean(day_scores)
            total_days_score += average_day_score
            total_days += 1

    if total_days > 0:
        return total_days_score / total_days
    else:
        return 0.0

def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f if line.strip()]

def compute_average_spatial_score(file_path):
    data = load_jsonl(file_path)
    total_score = 0
    count = 0

    for entry in data:
        try:
            score = calculate_spatial_score(entry)
            total_score += score
            count += 1
        except Exception as e:
            print(f"Skipping one plan due to error: {e}")
            continue

    return total_score / count if count > 0 else 0.0

def main():
    annotation_file = "/scratch/sg/Priyanshu/TripCraft-main/annotation_plan_3day_phi.jsonl"
    revised_file = "/scratch/sg/Priyanshu/TripCraft-main/revised_plan_3day_phi.jsonl"

    annotation_score = compute_average_spatial_score(annotation_file)
    revised_score = compute_average_spatial_score(revised_file)

    print(f"Average Spatial Score - Annotation Plan: {annotation_score:.4f}")
    print(f"Average Spatial Score - Revised Plan: {revised_score:.4f}")

if __name__ == "__main__":
    main()
"""
"""
import json
import numpy as np

# Parameters based on paper
d0 = 5000     # distance threshold in meters
lambda_ = 0.0002  # decay rate

def spatial_score(d):
    if d <= d0:
        return 1 - 0.5 * (d / d0)
    else:
        return 0.5 * np.exp(-lambda_ * (d - d0))

def calculate_avg_spatial_score(plan_obj):
    try:
        plan = plan_obj["plan"]
    except KeyError:
        return None  # plan missing

    poi_scores = []
    for day in plan:
        poi_str = day.get("point_of_interest_list", "")
        for segment in poi_str.split(";"):
            if "nearest transit:" in segment:
                try:
                    distance_part = segment.split("nearest transit:")[1].strip()
                    distance_str = distance_part.split(",")[-1].replace("m away.", "").strip()
                    distance = float(distance_str)
                    poi_scores.append(spatial_score(distance))
                except Exception:
                    continue  # skip malformed entries

    return np.mean(poi_scores) if poi_scores else None

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

def compute_Aspa(annotation_path, revised_path):
    anno_scores = []
    revised_scores = []

    for plan in load_jsonl(annotation_path):
        try:
            score = calculate_avg_spatial_score(plan["phi4_direct_og_sole-planning_results"])
            if score is not None:
                anno_scores.append(score)
        except Exception:
            continue

    for plan in load_jsonl(revised_path):
        try:
            score = calculate_avg_spatial_score(plan["phi4_direct_og_sole-planning_results"])
            if score is not None:
                revised_scores.append(score)
        except Exception:
            continue

    mean_anno = np.mean(anno_scores) if anno_scores else 0
    mean_revised = np.mean(revised_scores) if revised_scores else 0
    aspa = mean_anno - mean_revised

    print(f"Mean Annotation Spatial Score: {mean_anno:.4f}")
    print(f"Mean Revised Spatial Score:    {mean_revised:.4f}")
    print(f"Aspa (Spatial Adaptability):   {aspa:.4f}")

if __name__ == "__main__":
    annotation_path = "/scratch/sg/Priyanshu/TripCraft-main/annotation_plan_7day.jsonl"
    revised_path = "/scratch/sg/Priyanshu/TripCraft-main/revised_plan_7day_phi_normalized_5.jsonl"
    compute_Aspa(annotation_path, revised_path)
"""
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
    annotation_file = "/scratch/sg/Priyanshu/TripCraft-main/anno_plan_plan_level_gpt.jsonl"
    revised_file = "/scratch/sg/Priyanshu/TripCraft-main/qwen_plan_level_plan.jsonl"

    ann_score = compute_average_spatial_score(annotation_file)
    rev_score = compute_average_spatial_score(revised_file)
    delta_score = ann_score - rev_score

    print(f"Mean Annotation Spatial Score: {ann_score:.4f}")
    print(f"Mean Revised Spatial Score:    {rev_score:.4f}")
    print(f"Spatial Adaptability (Aspa):   {delta_score:.4f}")

if __name__ == "__main__":
    main()
