import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import time
import pickle
import sys
import os

DISQUALIFIED_COMPANIES = ["tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini"]
PREFERRED_LOCATIONS = ["pune", "noida"]
WELCOME_LOCATIONS = ["hyderabad", "mumbai", "delhi ncr", "delhi-ncr"]

def generate_reasoning(feat, sim_score, rank):
    reasons = []
    reasons.append(f"Ranked #{rank} with JD embedding similarity {sim_score:.2f}.")
    reasons.append(f"Has {feat['yoe']} YOE as {feat['current_title']}.")
    
    loc = str(feat.get('location', '')).lower()
    if any(pl in loc for pl in PREFERRED_LOCATIONS):
        reasons.append("Location is a primary preference.")
    elif any(wl in loc for wl in WELCOME_LOCATIONS):
        reasons.append("Location is in welcome zones.")
    else:
        reasons.append("Location is outside preferred zones.")
        
    reasons.append(f"Notice period: {feat['notice_period']} days.")
    reasons.append(f"Recruiter response rate: {feat['recruiter_rr']:.1%}.")
    
    if feat.get('is_pure_services'):
        reasons.append("[WARNING] Career entirely in pure-services firms.")
        
    if feat.get('is_hopper'):
        reasons.append("[WARNING] Title-hopper pattern detected (<18m avg).")
        
    if feat['plausibility_penalty'] > 0:
        reasons.append("[FLAG] Severe inconsistencies in profile timeline/skills detected.")
        
    if feat['title_relevance_multiplier'] < 0.5:
        reasons.append("[FLAG] Titles lack engineering/ML relevance, despite skills.")
        
    return " ".join(reasons)

def rank_candidates():
    start_time = time.time()
    
    # Disable network requests for sentence_transformers by forcing local only if already downloaded
    # (Setting local_files_only=True ensures it throws if network is needed, satisfying hackathon constraints)
    print("Loading models and precomputed data (local only)...")
    model = SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True)
    
    with open("job_description.txt", "r") as f:
        jd_text = f.read()
        
    jd_embedding = model.encode([jd_text], convert_to_numpy=True).astype('float32')
    
    index = faiss.read_index("faiss_index.bin")
    
    with open("precomputed_features.pkl", "rb") as f:
        features = pickle.load(f)
        
    print("Searching FAISS index...")
    # Retrieve top 2000 by raw similarity to process (ensure we have enough after filtering)
    k = min(2000, len(features))
    distances, indices = index.search(jd_embedding, k)
    
    results = []
    
    for i in range(k):
        idx = indices[0][i]
        dist = distances[0][i]
        sim_score = 1.0 / (1.0 + float(dist)) # Convert L2 to 0-1 similarity
        feat = features[idx]
        
        # 1. Experience band (5-9 years preferred)
        yoe = feat['yoe']
        if 5 <= yoe <= 9:
            exp_score = 1.0
        elif 4 <= yoe <= 10:
            exp_score = 0.5
        else:
            exp_score = 0.0
            
        # 2. Location
        loc = str(feat.get('location', '')).lower()
        if any(pl in loc for pl in PREFERRED_LOCATIONS):
            loc_score = 1.0
        elif any(wl in loc for wl in WELCOME_LOCATIONS):
            loc_score = 0.8
        else:
            loc_score = 0.5
            
        # 3. Pure Services Check
        career = feat.get('career_history', [])
        is_pure_services = False
        if len(career) > 0:
            has_product_exp = False
            for role in career:
                comp = str(role.get('company', '')).lower()
                if not any(dc in comp for dc in DISQUALIFIED_COMPANIES):
                    has_product_exp = True
                    break
            if not has_product_exp:
                is_pure_services = True
        feat['is_pure_services'] = is_pure_services
        
        # 4. Title hopper
        durations = [r.get('duration_months', 0) for r in career]
        avg_dur = sum(durations)/len(durations) if durations else 0
        is_hopper = len(durations) > 2 and avg_dur < 18
        feat['is_hopper'] = is_hopper
        
        # 5. Notice Period Multiplier
        np_days = feat['notice_period']
        if np_days <= 30:
            np_mult = 1.0
        elif np_days <= 60:
            np_mult = 0.8
        else:
            np_mult = 0.5
            
        # Combine Scores
        base_score = (sim_score * 0.4) + (exp_score * 0.2) + (loc_score * 0.1) + (feat['recruiter_rr'] * 0.3)
        
        # Apply Traps & Penalties
        final_score = base_score * feat['title_relevance_multiplier'] * np_mult
        
        if is_pure_services:
            final_score *= 0.5
        if is_hopper:
            final_score *= 0.7
            
        final_score -= feat['plausibility_penalty']
        
        if final_score < 0:
            final_score = 0.0
            
        results.append({
            "candidate_id": feat["candidate_id"],
            "score": final_score,
            "feat": feat,
            "jd_sim": sim_score
        })
        
    print("Sorting and formatting results...")
    # Sort by score desc, then candidate_id desc (deterministic tie-break)
    results.sort(key=lambda x: (x["score"], x["candidate_id"]), reverse=True)
    
    final_rows = []
    # Take exactly top 100
    for rank, res in enumerate(results[:100], 1):
        reasoning = generate_reasoning(res["feat"], res["jd_sim"], rank)
        final_rows.append({
            "candidate_id": res["candidate_id"],
            "rank": rank,
            "score": round(res["score"], 4),
            "reasoning": reasoning
        })
        
    df = pd.DataFrame(final_rows)
    df.to_csv("submission.csv", index=False)
    
    elapsed = time.time() - start_time
    # Simple memory estimate since we are constrained to 16GB. 
    # For a real env, psutil can be used, but since we are just reporting we can print the sizes.
    import sys
    print(f"Ranking step completed in {elapsed:.2f} seconds.")
    print("Output saved to submission.csv.")

if __name__ == "__main__":
    rank_candidates()
