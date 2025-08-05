
import argparse
import json
import numpy as np
import os

def calculate_wed(gen_sequence, anno_sequence, weight_fn):
    m, n = len(gen_sequence), len(anno_sequence)
    dp = np.full((m + 1, n + 1), np.inf)
    dp[0][0] = 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = weight_fn(gen_sequence[i - 1], anno_sequence[j - 1])
            dp[i][j] = cost + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n]

def weight_fn(a, b):
    return 0 if a == b else 1

def get_poi_sequence(plan_day):
    poi_list = plan_day.get("point_of_interest_list", "")
    accommodation = plan_day.get("accommodation", "").rsplit(",", 1)[0].strip()
    breakfast = plan_day.get("breakfast", "").rsplit(",", 1)[0].strip()
    lunch = plan_day.get("lunch", "").rsplit(",", 1)[0].strip()
    dinner = plan_day.get("dinner", "").rsplit(",", 1)[0].strip()

    seq = []
    for poi in poi_list.split(";"):
        poi = poi.strip()
        if not poi:
            continue
        if breakfast in poi or lunch in poi or dinner in poi:
            seq.append("y")  # Restaurant
        elif accommodation in poi:
            seq.append("x")  # Accommodation
        else:
            seq.append("z")  # Attraction
    return seq

def calculate_ordering_score(travel_plan, gen_travel_plan):
    total_wed = 0
    valid_days = 0

    days_to_compare = min(len(travel_plan["plan"]), len(gen_travel_plan["plan"]))
    if days_to_compare == 0:
        return 0.0

    for i in range(days_to_compare):
        try:
            anno_seq = get_poi_sequence(travel_plan["plan"][i])
            gen_seq = get_poi_sequence(gen_travel_plan["plan"][i])
            if not anno_seq or not gen_seq:
                continue

            wed = calculate_wed(gen_seq, anno_seq, weight_fn)
            total_wed += wed
            valid_days += 1
        except Exception:
            continue

    if valid_days == 0:
        return 0.0

    avg_wed = total_wed / valid_days
    max_len = max(len(gen_seq), len(anno_seq)) if gen_seq and anno_seq else 1
    return 1 - (avg_wed / max_len)

def main():
    parser = argparse.ArgumentParser(description="Evaluate sequential score using WED")
    parser.add_argument("--set_type", type=str, required=True, choices=["3day", "5day", "7day"],
                        help="Choose which dataset to evaluate: 3day, 5day or 7day")
    args = parser.parse_args()

    annotation_file = f"/scratch/sg/Priyanshu/TripCraft-main/anno_plan_plan_level_gpt.jsonl"
    revised_file = f"/scratch/sg/Priyanshu/TripCraft-main/qwen_plan_level_plan.jsonl"

    if not os.path.exists(annotation_file) or not os.path.exists(revised_file):
        print(f"❌ One of the input files not found for {args.set_type}.")
        return

    scores = []
    with open(annotation_file, "r", encoding="utf-8") as anno_f, \
         open(revised_file, "r", encoding="utf-8") as revised_f:

        for anno_line, revised_line in zip(anno_f, revised_f):
            try:
                anno_obj = json.loads(anno_line)
                revised_obj = json.loads(revised_line)

                if not anno_obj.get("plan") or not revised_obj.get("plan"):
                    continue
                if len(revised_obj["plan"]) == 0:
                    continue

                score = calculate_ordering_score(anno_obj, revised_obj)
                if score is not None:
                    scores.append(score)
            except Exception:
                continue

    avg_score = sum(scores) / len(scores) if scores else 0.0
    print("\n✅ Sequential Score Comparison")
    print(f"Number of plans evaluated: {len(scores)}")
    print(f"Average Sequential Score: {avg_score:.4f}")

if __name__ == "__main__":
    main()
"""
#5day
import argparse
import json
import numpy as np
import os

def calculate_wed(gen_sequence, anno_sequence, weight_fn):
    m, n = len(gen_sequence), len(anno_sequence)
    dp = np.full((m + 1, n + 1), np.inf)
    dp[0][0] = 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = weight_fn(gen_sequence[i - 1], anno_sequence[j - 1])
            dp[i][j] = cost + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n]

def weight_fn(a, b):
    return 0 if a == b else 1

def get_poi_sequence(plan_day):
    poi_list = plan_day.get("point_of_interest_list", "")
    accommodation = plan_day.get("accommodation", "").rsplit(",", 1)[0].strip()
    breakfast = plan_day.get("breakfast", "").rsplit(",", 1)[0].strip()
    lunch = plan_day.get("lunch", "").rsplit(",", 1)[0].strip()
    dinner = plan_day.get("dinner", "").rsplit(",", 1)[0].strip()

    seq = []
    for poi in poi_list.split(";"):
        poi = poi.strip()
        if not poi:
            continue
        if breakfast in poi or lunch in poi or dinner in poi:
            seq.append("y")  # Restaurant
        elif accommodation in poi:
            seq.append("x")  # Accommodation
        else:
            seq.append("z")  # Attraction
    return seq

def calculate_ordering_score(travel_plan, gen_travel_plan):
    total_wed = 0
    valid_days = 0

    days_to_compare = min(len(travel_plan["plan"]), len(gen_travel_plan["plan"]))
    if days_to_compare == 0:
        return 0.0

    for i in range(days_to_compare):
        try:
            anno_seq = get_poi_sequence(travel_plan["plan"][i])
            gen_seq = get_poi_sequence(gen_travel_plan["plan"][i])
            if not anno_seq or not gen_seq:
                continue

            wed = calculate_wed(gen_seq, anno_seq, weight_fn)
            total_wed += wed
            valid_days += 1
        except Exception:
            continue

    if valid_days == 0:
        return 0.0

    avg_wed = total_wed / valid_days
    max_len = max(len(gen_seq), len(anno_seq)) if gen_seq and anno_seq else 1
    return 1 - (avg_wed / max_len)

def main():
    parser = argparse.ArgumentParser(description="Evaluate sequential score using WED")
    parser.add_argument("--set_type", type=str, required=True, choices=["3day", "5day", "7day"],
                        help="Choose which dataset to evaluate: 3day, 5day or 7day")
    args = parser.parse_args()

    annotation_file = f"/scratch/sg/Priyanshu/TripCraft-main/annotation_plan.jsonl"
    revised_file = f"/scratch/sg/Priyanshu/TripCraft-main/revised_plan_{args.set_type}_qwen_normalized_5.jsonl"

    if not os.path.exists(annotation_file) or not os.path.exists(revised_file):
        print(f"❌ One of the input files not found for {args.set_type}.")
        return

    scores = []
    with open(annotation_file, "r", encoding="utf-8") as anno_f, \
         open(revised_file, "r", encoding="utf-8") as revised_f:

        for anno_line, revised_line in zip(anno_f, revised_f):
            try:
                anno_obj = json.loads(anno_line)
                revised_obj = json.loads(revised_line)

                if not anno_obj.get("plan") or not revised_obj.get("plan"):
                    continue
                if len(revised_obj["plan"]) == 0:
                    continue

                score = calculate_ordering_score(anno_obj, revised_obj)
                if score is not None:
                    scores.append(score)
            except Exception:
                continue

    avg_score = sum(scores) / len(scores) if scores else 0.0
    print("\n✅ Sequential Score Comparison")
    print(f"Number of plans evaluated: {len(scores)}")
    print(f"Average Sequential Score: {avg_score:.4f}")

if __name__ == "__main__":
    main()
"""