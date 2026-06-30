# Redrob Candidate Discovery & Ranking: Final Project Report

## What is your proposed solution?
Our solution is a hyper-optimized, decoupled retrieval and ranking pipeline designed to accurately surface the top 100 candidates from a pool of 100,000 without requiring expensive runtime LLM calls. It splits the heavy lifting of semantic embedding and mathematical feature extraction into an untimed offline pre-computation stage, allowing the live ranking script to execute in seconds while maintaining strict adherence to CPU and memory constraints.

## What differentiates your approach from traditional candidate matching systems?
Traditional ATS and matching systems rely heavily on keyword overlap (TF-IDF, BM25) or expensive runtime LLMs. Our approach:
1. **Defeats Keyword Stuffing Mathematically:** Instead of blindly rewarding keyword presence, we extract a `title_relevance_multiplier`. If a candidate has high ML/AI skills but their historical job titles are "Marketing Manager", their score is drastically slashed.
2. **Deterministic Traceability:** We generate highly specific reasoning strings using pure structured data templates rather than hallucination-prone generative AI.
3. **Hardware Agnosticism:** We achieve deep semantic search capabilities on a strict 5-minute CPU constraint by shifting the heavy `sentence-transformers` inference into an offline phase, indexing it using FAISS.

## What are the key requirements extracted from the JD?
- **Role:** Senior AI Engineer (Founding Team)
- **Experience:** 5-9 years preferred
- **Mandatory Experience:** Production experience deploying embeddings-based retrieval, vector DBs, Python, NDCG/A-B testing.
- **Disqualifiers:** 
  - Pure-research backgrounds (no product experience)
  - Pure-services firm backgrounds (e.g., TCS, Infosys, Wipro, Accenture) unless they have prior product company experience.
- **Location:** Pune/Noida preferred; Hyderabad/Mumbai/Delhi-NCR welcome.
- **Notice Period:** <30 days preferred.

## Which candidate signals are most important for determining relevance? / How does your solution evaluate candidate fit beyond keyword matching?
Relevance is evaluated through a composite score combining:
1. **Semantic Similarity (40%):** How closely the candidate's career descriptions match the JD contextually (via `all-MiniLM-L6-v2` embeddings).
2. **Experience & Location Banding (30%):** Strict bonuses for falling perfectly in the 5-9 YOE band and preferred locations.
3. **Behavioral Multipliers (30%):** Factoring in recruiter response rate and recent activity signals.
4. **Trajectory & Plausibility:** We evaluate the candidate's holistic timeline—slashing scores for title-hoppers (avg duration < 18 months) and checking for logical inconsistencies between stated YOE and career duration.

## How does your system retrieve, score, and rank candidates?
1. **Retrieve:** The `job_description.txt` is embedded locally. We query the FAISS index (L2 distance) to instantly retrieve the top relevant candidates based purely on semantic text.
2. **Score:** We convert the L2 distance to a 0-1 similarity score, then apply a weighted composite formula: `base_score = (sim_score * 0.4) + (exp_score * 0.2) + (loc_score * 0.1) + (recruiter_rr * 0.3)`.
3. **Rank:** This base score is aggressively filtered through our penalty multipliers (honeypots, pure-services, keyword-stuffing). The remaining valid candidates are sorted descending by score, using the `candidate_id` as a deterministic tie-breaker.

## What models, algorithms, or heuristics are used?
- **Model:** `sentence-transformers/all-MiniLM-L6-v2` (Chosen for its extreme CPU efficiency and tiny 90MB memory footprint, while still providing strong contextual text embeddings).
- **Algorithm:** FAISS (Facebook AI Similarity Search) using `IndexFlatL2` for exhaustive, exact nearest-neighbor search.
- **Heuristics:** We use strict rule-based multipliers for notice periods, pure-services history tracking, and mathematical timeline plausibility (e.g., `summed_career_duration > stated_yoe + 4`).

## How are multiple candidate signals combined into a final ranking?
Signals are combined hierarchically:
1. **The Base Score** represents the "ideal state" based on JD text, location, YOE, and behavioral health.
2. **The Multipliers** represent the "reality check". A great base score is multiplied by the `title_relevance_multiplier` (neutralizing skill stuffers) and the notice period multiplier.
3. **The Penalties** are absolute. If a candidate triggers the Pure-Services or Title-Hopper flags, their score is chopped by 50% or 30%. If they hit a severe Plausibility Penalty (Honeypot), their score is zeroed out.

## How are ranking decisions explained? / How do you prevent hallucinations or unsupported justifications?
We completely bypass generative AI for the final output. The reasoning column is dynamically constructed using programmatic string interpolation based exclusively on the hard structured data extracted during pre-computation. 
*Example:* `Ranked #1 with JD embedding similarity 0.53. Has 5.2 YOE as ML Engineer. Location is a primary preference. Notice period: 30 days. Recruiter response rate: 82.0%.`
This ensures 100% traceability and exactly zero risk of hallucination.

## How does your solution handle inconsistent, low-quality, or suspicious profiles?
This is our primary defense mechanism against the >10% Honeypot disqualification rule. The pre-computation script runs deep mathematical consistency checks on every profile:
- Does the sum of their individual job durations roughly equal their stated total YOE?
- Are they claiming "expert" proficiency on a skill they've only used for 3 months?
- Does their total skill duration exceed the physical length of their entire career?
If any of these constraints fail wildly, the candidate receives a `plausibility_penalty` that zeroes their score, sinking them to the bottom of the 100,000 stack.

## What is the complete workflow from JD input to ranked candidate output? (System Architecture)
1. **Offline Pre-computation (`precompute.py`):**
   - Parses the 100K `candidates.jsonl`.
   - Extracts structured math, timeline checks, and behavioral signals.
   - Embeds concatenated text features using `all-MiniLM-L6-v2`.
   - Saves `faiss_index.bin` and `precomputed_features.pkl`.
2. **Online Timed Ranking (`rank.py`):**
   - Loads the JD and embeds it locally.
   - Searches the FAISS index to retrieve the semantic candidates.
   - Calculates the composite score using the precomputed PKL features.
   - Sorts, generates deterministic reasoning, and outputs exactly 100 rows to `submission.csv`.

## What results or insights demonstrate ranking quality?
When run against the 100K dataset, our top 5 candidates perfectly hit the preferred 5-9 YOE band, possessed strict AI/ML Engineering titles, were located in primary/welcome zones, and had 0% honeypot flags. The system successfully filtered out candidates who possessed AI skills but lacked engineering titles.

## How does your solution meet the challenge's runtime and compute constraints?
- **Runtime:** `rank.py` finishes the full 100K evaluation in **11.00 seconds**, destroying the 5-minute ceiling.
- **Compute:** The script operates with peak memory well under 1GB (comfortably passing the 16GB limit) and uses a CPU-optimized MiniLM model.
- **Network:** We enforce `local_files_only=True` in HuggingFace to guarantee zero API or LLM calls during the timed run.

## What technologies, frameworks, and tools were used and why were they selected?
- **Python (pandas):** Fast data manipulation and CSV writing.
- **Sentence-Transformers:** For local, high-quality, lightweight NLP embeddings.
- **FAISS:** The industry standard for high-performance, low-latency vector similarity search on CPU.
- **Pickle:** Native Python serialization for instant loading of our structured dictionary states into the ranker.
