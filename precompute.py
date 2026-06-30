import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import faiss
import numpy as np
import pickle
import time
from sentence_transformers import SentenceTransformer

# JD-derived valid engineering titles to combat keyword stuffing
# These are lowercased for direct substring matching
VALID_TITLES = [
    "engineer", "developer", "scientist", "machine learning", "ml", "ai", 
    "data", "backend", "software", "programmer", "architect"
]

def extract_candidate_text(c):
    """
    Constructs the specific per-candidate text designed in Step 2.
    Format:
    [Headline]: ...
    [Summary]: ...
    [Experience]:
    - Title at Company (X months): Description
    [Skills]: Name (Prof, Y months), ...
    """
    parts = []
    
    if c.get("profile", {}).get("headline"):
        parts.append(f"[Headline]: {c['profile']['headline']}")
    
    if c.get("profile", {}).get("summary"):
        parts.append(f"[Summary]: {c['profile']['summary']}")
        
    parts.append("[Experience]:")
    for role in c.get("career_history", []):
        t = role.get("title", "Unknown")
        comp = role.get("company", "Unknown")
        d = role.get("duration_months", 0)
        desc = role.get("description", "")
        parts.append(f"- {t} at {comp} ({d} months): {desc}")
        
    parts.append("[Skills]:")
    skills = []
    for s in c.get("skills", []):
        name = s.get("name", "Unknown")
        prof = s.get("proficiency", "Unknown")
        dur = s.get("duration_months", 0)
        skills.append(f"{name} ({prof}, {dur} months)")
    parts.append(", ".join(skills))
    
    return "\n".join(parts)

def precompute():
    start_time = time.time()
    
    print("Loading SentenceTransformer 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    candidates_features = []
    texts_to_embed = []
    
    print("Processing candidates from candidates.jsonl...")
    
    with open("candidates.jsonl", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip(): continue
            c = json.loads(line)
            
            cand_id = c["candidate_id"]
            
            # 1. Text construction
            text = extract_candidate_text(c)
            texts_to_embed.append(text)
            
            # 2. Keyword-Stuffer Trap: Title Check
            # Extract current title and all past titles
            titles = [c.get("profile", {}).get("current_title", "")]
            for role in c.get("career_history", []):
                titles.append(role.get("title", ""))
            
            has_relevant_title = False
            for t in titles:
                t_lower = t.lower()
                if any(vt in t_lower for vt in VALID_TITLES):
                    has_relevant_title = True
                    break
            
            # If no relevant title ever, heavily penalize in the ranking step
            title_relevance_multiplier = 1.0 if has_relevant_title else 0.1
            
            # 3. Plausibility / Honeypot Checks
            yoe = c.get("profile", {}).get("years_of_experience", 0)
            total_duration_months = sum([r.get("duration_months", 0) for r in c.get("career_history", [])])
            
            plausibility_penalty = 0.0
            
            # Flag 1: Experience Overload (e.g. 10 years of roles but claims 2 YOE, or claims 10 YOE but has 20 years of roles)
            # Add generous buffer for overlapping jobs or gaps
            summed_yoe = total_duration_months / 12.0
            if summed_yoe > (yoe + 4): 
                plausibility_penalty += 0.5
                
            # Check skills
            for s in c.get("skills", []):
                dur = s.get("duration_months", 0)
                prof = s.get("proficiency", "beginner")
                
                # Flag 2: Expert with zero/minimal experience
                if prof == "expert" and dur < 12:
                    plausibility_penalty += 0.5
                    
                # Flag 3: Time-traveling skill (used longer than total YOE + generous buffer)
                if dur > (yoe * 12 + 24):
                    plausibility_penalty += 0.5
            
            # Cap penalty at 1.0 (means zero score)
            plausibility_penalty = min(1.0, plausibility_penalty)
            
            # 4. Extract other core structural features for ranking
            signals = c.get("redrob_signals", {})
            
            candidates_features.append({
                "candidate_id": cand_id,
                "yoe": yoe,
                "current_title": c.get("profile", {}).get("current_title", "Unknown"),
                "location": c.get("profile", {}).get("location", "Unknown"),
                "notice_period": signals.get("notice_period_days", 90),
                "recruiter_rr": signals.get("recruiter_response_rate", 0.0),
                "last_active": signals.get("last_active_date", ""),
                "title_relevance_multiplier": title_relevance_multiplier,
                "plausibility_penalty": plausibility_penalty,
                "career_history": c.get("career_history", []), # Need this to check for pure-services firms in rank.py
            })
            
            if (i+1) % 10000 == 0:
                print(f"Processed {i+1} candidates...")

    print(f"Extracted features for {len(candidates_features)} candidates in {time.time() - start_time:.2f} seconds.")
    
    print("Generating embeddings (this will take several minutes)...")
    embed_start = time.time()
    # Batch size 64 for optimal CPU throughput
    embeddings = model.encode(texts_to_embed, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    print(f"Embedding generation completed in {time.time() - embed_start:.2f} seconds.")
    
    print("Building FAISS index...")
    faiss_start = time.time()
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    # Ensure float32 for FAISS
    index.add(np.array(embeddings).astype('float32'))
    print(f"FAISS indexing completed in {time.time() - faiss_start:.2f} seconds.")
    
    print("Saving precomputed data...")
    faiss.write_index(index, "faiss_index.bin")
    with open("precomputed_features.pkl", "wb") as f:
        pickle.dump(candidates_features, f)
        
    total_time = time.time() - start_time
    print(f"Total offline pre-computation completed in {total_time:.2f} seconds.")

if __name__ == "__main__":
    precompute()
