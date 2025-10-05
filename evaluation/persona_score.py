
import json
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_bert_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze(0).numpy()

def compute_persona_score(entry, bert_model, bert_tokenizer):
    # Case 1: entry is a list → unwrap it
    if isinstance(entry, list):
        entry = entry[0] if entry else {}

    # Case 2: dict has inner wrapped key
    if "phi4_direct_og_sole-planning_results" in entry:
        travel_plan = entry["phi4_direct_og_sole-planning_results"]
    else:
        travel_plan = entry  # already flattened / normalized

    persona = travel_plan.get("persona")
    if not persona:
        print("Warning: 'persona' key missing. Skipping.")
        return 0.0

    # Handle persona being dict OR string
    persona_components = {}
    if isinstance(persona, dict):
        # Structured form
        for key in ["Traveler Type", "Purpose of Travel", "Spending Preference", "Location Preference"]:
            persona_components[key] = persona.get(key, "")
    elif isinstance(persona, str):
        # String form, parse manually
        for key in ["Traveler Type", "Purpose of Travel", "Spending Preference", "Location Preference"]:
            start_idx = persona.find(key + ":")
            if start_idx == -1:
                persona_components[key] = ""
                continue
            start_idx += len(key) + 1
            end_idx = persona.find(";", start_idx)
            value = persona[start_idx:end_idx].strip() if end_idx != -1 else persona[start_idx:].strip()
            persona_components[key] = value
    else:
        print(f"Unexpected persona format: {type(persona)}. Skipping.")
        return 0.0

    # Embed persona components
    persona_embeddings = {
        key: get_bert_embedding(value, bert_tokenizer, bert_model)
        for key, value in persona_components.items()
    }

    total_score = 0
    total_pois = 0

    for day in travel_plan.get("plan", []):
        poi_data = day.get("point_of_interest_list", None)
        if not poi_data:
            continue

        # Normalize to list of strings
        if isinstance(poi_data, str):
            if poi_data.strip() == "-":
                continue
            poi_list = [p.strip() for p in poi_data.split(";") if p.strip()]
        elif isinstance(poi_data, list):
            poi_list = [str(p).strip() for p in poi_data if str(p).strip() != "-"]
        else:
            print(f"Unexpected POI format: {type(poi_data)}. Skipping.")
            continue

        for poi in poi_list:
            if "stay" in poi:
                poi_name = poi.split("stay")[0].strip()[:-1]
            else:
                poi_name = poi.split("visit")[0].strip()[:-1]

            poi_embedding = get_bert_embedding(poi_name, bert_tokenizer, bert_model)

            poi_score = 0
            for key, persona_embedding in persona_embeddings.items():
                poi_score += cosine_similarity([persona_embedding], [poi_embedding])[0][0]
            poi_score /= len(persona_components)

            total_score += poi_score
            total_pois += 1

    return total_score / total_pois if total_pois > 0 else 0.0



def load_jsonl(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

def calculate_average_persona_score(jsonl_path):
    plans = load_jsonl(jsonl_path)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    scores = []
    for plan in plans:
        score = compute_persona_score(plan, model, tokenizer)
        scores.append(score)

    avg_score = np.mean(scores)
    return avg_score

if __name__ == "__main__":
    anno_path = "annotation_plan_7day.jsonl"
    revised_path = "7day_phi4_revised_clean.jsonl"

    print("Calculating Persona Scores...")
    anno_avg = calculate_average_persona_score(anno_path)
    revised_avg = calculate_average_persona_score(revised_path)

    print(f"Average Persona Score - Annotation Plan: {anno_avg:.4f}")
    print(f"Average Persona Score - Revised Plan:    {revised_avg:.4f}")
