# Tech Stack Recommender
### DecodeLabs Industrial Training — AI Track | Project 3 | Batch 2026

A content-based recommendation engine that maps your skills to the most relevant tech career paths. Built from scratch using **TF-IDF vectorization** and **Cosine Similarity** — no ML black boxes, no magic. Every step is intentional and explainable.

---

## What It Does

You enter skills you already have. The system scores every job role in its catalogue against those skills using real similarity math, ranks them, and returns the **Top 3 best-fit career paths** with a percentage match score and the specific skills that drove the match.

```
Input:  "Python, SQL, Machine Learning, Statistics, Pandas"
Output: #1 Data Scientist       — 62.3% match
        #2 ML Engineer          — 31.6% match
        #3 NLP Engineer         — 29.3% match
```

---

## How It Works — The Full Pipeline

The system follows a strict **4-step pipeline** (same architecture used by Netflix, LinkedIn, Spotify):

```
[User Input] → Ingestion → Scoring → Sorting → Filtering → [Top-N Results]
```

### Step 1 · Ingestion
Raw user input is normalized — abbreviations expanded (`ML` → `Machine Learning`, `k8s` → `Kubernetes`), casing standardized, whitespace stripped. The cleaned string becomes the **user profile document**.

### Step 2 · Scoring
The user profile is converted into a **TF-IDF vector** using the same vocabulary fitted on the job roles catalogue. Cosine similarity is then computed between the user's vector and every job role vector simultaneously.

### Step 3 · Sorting
All scores are sorted in descending order using `np.argsort()[::-1]` — highest similarity first.

### Step 4 · Filtering
The sorted list is truncated at `top_n` (default: 3) to prevent choice overload. Only the highest-scoring roles are shown.

---

## The Math Behind It

### TF-IDF (Why Not Just Count Words?)

Simple binary matching (skill present = 1, absent = 0) treats `"Python"` — which appears in 12 of 15 roles — the same as `"NLP"` — which appears in only 2. That's useless.

TF-IDF fixes this:

```
TF(skill, role)  = count of skill in this role / total skills in this role
IDF(skill)       = log( total roles / roles containing this skill )
Weight           = TF × IDF
```

Rare, specific skills like `"Kubernetes"` or `"NLP"` get **high weights**. Common skills like `"Python"` or `"Git"` get **low weights**. The similarity math now reflects meaningful signal.

### Cosine Similarity (Why Not Euclidean Distance?)

Euclidean distance is sensitive to **magnitude** — a job role description with 20 skills listed would always seem farther away than one with 5, even if the match pattern is identical.

Cosine similarity measures the **angle** between vectors, not the distance:

```
cos(θ) = (A · B) / (||A|| × ||B||)
```

- `1.0` → perfect alignment
- `0.5` → moderate overlap
- `0.0` → no shared features

Because TF-IDF values are non-negative, our scores naturally fall in `[0, 1]` — which reads cleanly as a percentage match.

---

## Libraries Used

| Library | Version | Why |
|---|---|---|
| `scikit-learn` | ≥ 1.0 | `TfidfVectorizer` + `cosine_similarity` |
| `numpy` | ≥ 1.21 | Array math, `argsort` for sorting |
| `pandas` | ≥ 1.3 | Optional — for loading job roles from CSV |

---

## Installation

```bash
pip install scikit-learn numpy pandas
```

No other dependencies. Runs entirely in the terminal.

---

## Running the Project

```bash
python tech_stack_recommender.py
```

You'll be prompted to enter skills:

```
Your skills: Python, SQL, Machine Learning, Statistics
```

**Minimum 3 skills required** — anything fewer doesn't give the algorithm enough signal for meaningful matching.

Other commands:
- `explore` — list all 15 job roles and their skill tags
- `quit` — exit the program

---

## Project Structure

```
tech_stack_recommender.py
│
├── SKILL_ALIASES {}           # Maps abbreviations to canonical skill names
├── JOB_ROLES []               # The catalogue — 15 roles, each with a skills string
│
├── normalize_skills()         # Step 1: Clean and standardize user input
├── build_tfidf_model()        # Fits TfidfVectorizer on the job roles corpus
├── get_recommendations()      # Steps 2-4: Score, sort, filter
├── cold_start_recommendations() # Fallback when no input given
├── display_results()          # Output formatting
└── main()                     # Interaction loop
```

---

## Key Syntax & Concepts Explained

### `TfidfVectorizer`
```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(analyzer='word', sublinear_tf=True)
item_matrix = vectorizer.fit_transform(corpus)
# fit_transform = learn vocabulary + convert to vectors (in one call)
# sublinear_tf=True → uses 1 + log(TF) instead of raw TF, keeps values comparable
```

### `vectorizer.transform()` vs `fit_transform()`
```python
# fit_transform → learns vocabulary AND converts (used on job roles — once)
item_matrix = vectorizer.fit_transform(corpus)

# transform only → uses existing vocabulary (used on user input — never refit)
user_vector = vectorizer.transform([user_skills_clean])
# Note: input must be a LIST even if it's a single string
```

### `cosine_similarity`
```python
from sklearn.metrics.pairwise import cosine_similarity

scores = cosine_similarity(user_vector, item_matrix)[0]
# Returns a 2D array — [0] extracts the first (only) row
# scores[i] = similarity between user and job_roles[i]
```

### `np.argsort()[::-1]`
```python
import numpy as np

scores = np.array([0.21, 0.87, 0.34])
ranked = np.argsort(scores)[::-1]
# np.argsort → [0, 2, 1]  (indices that sort ascending)
# [::-1]     → [1, 2, 0]  (reversed = descending)
# job_roles[ranked[0]] = the best match
```

### List Comprehension (corpus building)
```python
corpus = [role["skills"] for role in JOB_ROLES]
# Same as:
corpus = []
for role in JOB_ROLES:
    corpus.append(role["skills"])
```

---

## The Cold Start Problem

**Symptom:** A brand-new user enters nothing. Their vector = all zeros. Cosine similarity against a zero vector = 0 for everything — the system can't distinguish any role from any other.

**Solution implemented:** A trending fallback — when no input is given, the system returns a hardcoded list of globally popular roles instead of running the similarity math. In a production system this would pull from real interaction logs.

Two types of cold start exist:
- **User Cold Start** — new user, no profile → solved with fallback / onboarding survey
- **Item Cold Start** — new job role with no tags → content-based filtering handles this naturally (add tags, it immediately becomes recommendable)

---

## Edge Cases

| Input | What Happens | Why It Matters |
|---|---|---|
| `"ML, DL, k8s"` | Aliases expand before vectorizing | Tests normalization layer |
| `"Python"` (1 skill) | Rejected — minimum 3 required | Validation before algorithm runs |
| `"Blockchain, Web3, Solidity"` | Near-zero scores everywhere | Skills not in vocabulary → vector ≈ 0 |
| `"Python, React, Docker, Kubernetes, ML"` | Cross-domain input | Scores compress — tests generalist handling |
| `"Python, Python, Python"` | TF increases for repeated terms | Tests TF weighting behavior |
| Empty input | Cold start fallback triggers | Tests the zero-vector edge case |

---

## Difficulties & Limitations

**Vocabulary rigidity** — TF-IDF requires exact vocabulary matches. `"Machine Learning"` and `"ML"` are treated as completely different tokens unless the alias dictionary handles them. In production, this is solved with word embeddings (e.g. Sentence-BERT) which understand meaning, not just exact words.

**Dataset quality** — the recommendations are only as good as the skill tags in `JOB_ROLES`. If a role is missing key skills, it will score poorly even when it's the right match. Garbage in, garbage out.

**Score compression at high input density** — when the user enters 10+ skills from multiple domains, most roles get moderate scores and the rankings become less decisive. The top result is still correct, but margins narrow.

**No feedback loop** — the system doesn't learn. If the user says "this recommendation was wrong," nothing updates. A production system would use ratings to adjust weights over time.

---

## Extending This Project

```python
# 1. Load job roles from CSV instead of hardcoding
import pandas as pd
df = pd.read_csv('raw_skills.csv')
JOB_ROLES = df.apply(
    lambda row: {"title": row["role"], "skills": row["skills"]}, axis=1
).tolist()

# 2. Change how many results are returned
results = get_recommendations(..., top_n=5)  # Show top 5 instead of 3

# 3. Upgrade to semantic similarity (understands meaning, not just exact words)
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')
# Replace TF-IDF vectors with model.encode() outputs
```

---

## Sample Output

```
============================================================
   🤖 TECH STACK RECOMMENDER — DecodeLabs Project 3
   Powered by TF-IDF + Cosine Similarity
============================================================

[SYSTEM] Model loaded. Vocabulary size: 93 unique skill terms.
[SYSTEM] Catalogue: 15 job roles available.

Your skills: Python, Machine Learning, Deep Learning, TensorFlow

[SYSTEM] Normalized your input to: 'Python Machine Learning Deep Learning Tensorflow'

============================================================
  🎯 TOP CAREER PATH RECOMMENDATIONS
============================================================

  #1 — Machine Learning Engineer
      Match Score: [█████████████████████░░░░░░░░░] 72.4%
      Matched Skills: deep learning, learning, machine, python, tensorflow

  #2 — AI Research Scientist
      Match Score: [████████████████░░░░░░░░░░░░░░] 55.1%
      Matched Skills: deep learning, learning, machine, python

  #3 — NLP Engineer
      Match Score: [████████████░░░░░░░░░░░░░░░░░░] 41.3%
      Matched Skills: deep learning, learning, machine, python

============================================================
```

---

## Concepts Demonstrated

- Content-Based Filtering
- TF-IDF Vectorization
- Cosine Similarity
- Vector Space Models
- Cold Start Problem & mitigation strategies
- Input normalization and alias resolution
- Top-N ranking pipeline

---

*DecodeLabs Industrial Training Kit · Batch 2026 · AI Track · Project 3*
