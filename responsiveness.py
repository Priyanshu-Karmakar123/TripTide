import pandas as pd
import json
import argparse

def is_mitigated(annotation_plan, revised_plan):
    try:
        ann_days = annotation_plan.get("plan", [])
        rev_days = revised_plan.get("plan", [])

        # If number of days mismatch, it's considered mitigated
        if len(ann_days) != len(rev_days):
            return 1

        for day_ann, day_rev in zip(ann_days, rev_days):
            poi_ann = day_ann.get("point_of_interest_list", "").strip()
            poi_rev = day_rev.get("point_of_interest_list", "").strip()

            if poi_ann != poi_rev:
                return 1  # Mitigated due to difference

        return 0  # No difference found
    except Exception as e:
        print(f"Error comparing plans: {e}")
        return 0

def main():
    parser = argparse.ArgumentParser(description="Check if disruption has been mitigated.")
    parser.add_argument('--csv_file', type=str, required=True, help='Path to the CSV file (3-day, 5-day, or 7-day)')
    args = parser.parse_args()

    df = pd.read_csv(args.csv_file)

    mitigated_flags = []
    for idx, row in df.iterrows():
        try:
            ann_plan = json.loads(row["annotation_plan"])
            revised_plan = json.loads(row["revised_plan"])
            flag = is_mitigated(ann_plan, revised_plan)
            mitigated_flags.append(flag)
        except Exception as e:
            print(f"Row {idx} failed: {e}")
            mitigated_flags.append(0)

    df["mitigated"] = mitigated_flags
    total = len(mitigated_flags)
    mitigated = sum(mitigated_flags)
    mitigation_rate = mitigated / total if total > 0 else 0.0

    print(f"\n✅ Mitigation rate: {mitigated} / {total} = {mitigation_rate:.4f}")


if __name__ == "__main__":
    main()
