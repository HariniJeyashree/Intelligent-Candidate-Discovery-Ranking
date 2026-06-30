# Redrob Intelligent Candidate Discovery & Ranking Challenge

This repository contains a standalone submission for the Redrob Intelligent Candidate Discovery & Ranking Challenge.

## Architecture & Workflow

The solution is strictly divided into two phases to meet the hard constraints (<=5 minutes wall-clock, <=16GB RAM, CPU-only, zero network calls during ranking).

### 1. Offline Pre-computation (`precompute.py`)
This script processes the raw `candidates.jsonl` dataset and performs time-intensive operations. It does not run under the 5-minute limit.
- **Embeddings:** Uses a local `all-MiniLM-L6-v2` SentenceTransformer to generate 384-dimensional embeddings for candidate profiles (combining headline, summary, career history, and skills).
- **Indexing:** Builds a local FAISS CPU index for extremely fast retrieval.
- **Feature Extraction:** Precomputes structured features and flags directly from the schema:
  - Experience band fit (5-9 years).
  - Pure-services company disqualification (flags candidates from TCS, Infosys, etc.).
  - Title-hopper detection (flags candidates with low average tenure).
  - Plausibility & honeypot risk checks (flags unrealistic skill duration or year-of-experience inconsistencies).
  - Behavioral availability multipliers (combines recruiter response, notice period, and open-to-work flags).

### 2. Fast Ranking Step (`rank.py`)
This script executes under strict constraints.
- Loads the precomputed FAISS index and features (from `faiss_index.bin` and `precomputed_features.pkl`).
- Embeds the `job_description.txt` locally.
- Performs an ANN search to retrieve candidate matches based on semantic similarity.
- Computes a final composite score by combining the similarity score with the precomputed behavioral multipliers and penalties.
- Generates a templated `reasoning` column directly from data fields, guaranteeing zero hallucinations.
- Outputs exactly 100 rows to `submission.csv` in less than 5 minutes.

## Setup & Reproduction

1. **Install Requirements:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Pre-computation (Untimed):**
   Ensure `candidates.jsonl` and `job_description.txt` are in the directory.
   ```bash
   python precompute.py
   ```

3. **Ranking (Timed):**
   Run the ranking script to produce `submission.csv`.
   ```bash
   python rank.py
   ```

4. **Validation:**
   ```bash
   python validate_submission.py submission.csv
   ```
