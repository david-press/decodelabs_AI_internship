"""
=============================================================================
PROJECT 3: AI RECOMMENDATION LOGIC — TECH STACK RECOMMENDER
DecodeLabs Industrial Training Kit | Batch 2026
=============================================================================


# =============================================================================
# SECTION 1: IMPORTS — WHAT LIBRARIES DO WE NEED?
# =============================================================================

# numpy: Gives us powerful math tools. We use it to do vector math (dot products, norms).
# Think of it as a calculator that can handle lists of numbers at once.
import numpy as np

# sklearn (scikit-learn): The most popular Python Machine Learning library.
# We import two specific tools from it:
#   - TfidfVectorizer: Converts text (like skill tags) into TF-IDF weighted number vectors
#   - cosine_similarity: Calculates the cosine similarity score between two vectors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =============================================================================
# SECTION 2: THE DATASET — OUR "ITEMS" (Job Roles)
# =============================================================================
# This is our catalogue of job roles. Each role has a name and a list of skills.
# In recommendation systems, these are the "items" we recommend from.
#
# KEY INSIGHT: Item skills and user skills must use the SAME vocabulary.
# If the dataset says "Machine Learning" but the user types "ML", they won't match.
# We handle this with a normalization function below.

JOB_ROLES = [
    {
        "title": "Data Scientist",
        "skills": "Python SQL Machine Learning Statistics Data Analysis Pandas NumPy Scikit-Learn Visualization"
    },
    {
        "title": "Machine Learning Engineer",
        "skills": "Python Machine Learning Deep Learning TensorFlow PyTorch Neural Networks Model Deployment Docker"
    },
    {
        "title": "Backend Developer",
        "skills": "Python Java SQL APIs REST Databases Docker Microservices Node Git"
    },
    {
        "title": "Frontend Developer",
        "skills": "JavaScript HTML CSS React TypeScript UI Design REST APIs Git Figma"
    },
    {
        "title": "DevOps Engineer",
        "skills": "Docker Kubernetes AWS Cloud CI/CD Linux Automation Terraform Monitoring Git"
    },
    {
        "title": "Cloud Architect",
        "skills": "AWS Azure Cloud Networking Security Terraform Kubernetes Docker Architecture DevOps"
    },
    {
        "title": "Data Engineer",
        "skills": "Python SQL Spark ETL Kafka Airflow Data Pipelines Cloud Databases Big Data"
    },
    {
        "title": "Cybersecurity Analyst",
        "skills": "Networking Security Linux Penetration Testing Firewalls Cryptography Python Monitoring Compliance"
    },
    {
        "title": "AI Research Scientist",
        "skills": "Python Deep Learning Neural Networks Mathematics Statistics Research TensorFlow PyTorch NLP"
    },
    {
        "title": "Full Stack Developer",
        "skills": "JavaScript Python React Node SQL HTML CSS REST APIs Docker Git"
    },
    {
        "title": "Business Intelligence Analyst",
        "skills": "SQL Python Tableau Power BI Data Analysis Statistics Excel Visualization Reporting"
    },
    {
        "title": "Mobile App Developer",
        "skills": "Swift Kotlin React Native JavaScript APIs Mobile UI Design Git Firebase"
    },
    {
        "title": "NLP Engineer",
        "skills": "Python NLP Deep Learning Transformers BERT Hugging Face Text Processing Machine Learning"
    },
    {
        "title": "Robotics Engineer",
        "skills": "Python C++ ROS Robotics Control Systems Embedded Systems Sensors Automation Mathematics"
    },
    {
        "title": "Database Administrator",
        "skills": "SQL PostgreSQL MySQL MongoDB Optimization Backup Security Indexing Performance Tuning"
    },
]


# =============================================================================
# SECTION 3: SKILL NORMALIZATION — FIXING VOCABULARY MISMATCHES
# =============================================================================
# Problem: The user might type "ML" but our dataset says "Machine Learning".
# Solution: We build a dictionary of common abbreviations and map them to the
#           exact words our dataset uses. This is called "normalization".

SKILL_ALIASES = {
    "ml": "Machine Learning",
    "dl": "Deep Learning",
    "ai": "Machine Learning",
    "js": "JavaScript",
    "ts": "TypeScript",
    "k8s": "Kubernetes",
    "tf": "TensorFlow",
    "nlp": "NLP",
    "db": "Databases",
    "devops": "DevOps",
    "ci/cd": "CI/CD",
    "aws": "AWS",
    "api": "APIs",
    "react": "React",
    "node": "Node",
}


def normalize_skills(raw_input: str) -> str:
    """
    Takes raw user input (e.g. "python, ML, JS")
    Returns cleaned, normalized skill string (e.g. "Python Machine Learning JavaScript")

    HOW IT WORKS:
    1. Split the input by commas
    2. Strip extra whitespace from each skill
    3. Check if the lowercased version is in our alias dictionary → replace it
    4. Otherwise, title-case the skill (makes "python" → "Python")
    5. Join everything back into one string
    """
    # Split by comma, strip whitespace, filter out empty strings
    raw_skills = [s.strip() for s in raw_input.split(",") if s.strip()]

    normalized = []
    for skill in raw_skills:
        # Check alias dict using lowercase version
        lower_skill = skill.lower()
        if lower_skill in SKILL_ALIASES:
            normalized.append(SKILL_ALIASES[lower_skill])  # Use the mapped full name
        else:
            normalized.append(skill.title())  # Title-case it: "python" → "Python"

    # Join all skills into one string (the TfidfVectorizer expects a string, not a list)
    return " ".join(normalized)


# =============================================================================
# SECTION 4: BUILDING THE TF-IDF VECTOR SPACE
# =============================================================================
# This is the HEART of content-based filtering.
#
# WHAT IS TF-IDF?
#   - TF (Term Frequency): How often does a skill appear in ONE job role's description?
#     A skill that appears multiple times in one role is probably important to it.
#     Formula: TF = count of term in document / total terms in document
#
#   - IDF (Inverse Document Frequency): How rare is this skill across ALL job roles?
#     If "Python" appears in EVERY role, it's not very discriminating. Lower weight.
#     If "NLP" only appears in 2 roles, it's specific and powerful. Higher weight.
#     Formula: IDF = log(Total Documents / Documents containing the term)
#
#   - TF-IDF Weight = TF × IDF
#
# WHAT IS A VECTOR?
#   - A vector is just a list of numbers. 
#   - Imagine we have 50 unique skills across all roles. Each role becomes a list of 50 numbers.
#   - Each number represents how "important" that skill is to that role.
#   - Example: [0.0, 0.45, 0.0, 0.72, 0.0, ...] — non-zero means that skill matters here
#
# WHY NOT JUST COUNT WORDS (BINARY 0/1)?
#   - Binary matching treats "Python" (everywhere) same as "NLP" (rare and specific)
#   - TF-IDF gives HIGHER weight to rare, specific skills — much smarter!

def build_tfidf_model(job_roles: list) -> tuple:
    """
    Takes the list of job roles.
    Returns: (vectorizer, item_matrix)
        - vectorizer: the fitted TF-IDF model (knows the vocabulary)
        - item_matrix: matrix where each row is a job role's TF-IDF vector
    """

    # Extract all skill strings from our dataset
    # We get a list like: ["Python SQL Machine Learning...", "Python Machine Learning Deep Learning...", ...]
    corpus = [role["skills"] for role in job_roles]

    # TfidfVectorizer: This is the tool that does all the heavy lifting.
    # It reads the corpus, learns the vocabulary, and converts text to TF-IDF vectors.
    #
    # Parameters explained:
    #   analyzer='word'    → treat each individual word as a token (not letters or n-grams)
    #   token_pattern=...  → what counts as a "word"? We use r"(?u)\b[a-zA-Z][a-zA-Z+#./0-9-]+\b"
    #                         to handle words like "CI/CD", "C++", "TensorFlow" properly
    #   sublinear_tf=True  → use log(TF) instead of raw TF — dampens the effect of very
    #                         frequent terms, keeping values comparable
    vectorizer = TfidfVectorizer(
        analyzer='word',
        sublinear_tf=True  # log normalization for TF
    )

    # fit_transform does TWO things in one call:
    #   fit():       Learn the vocabulary from the corpus (discovers all unique skills)
    #   transform(): Convert each document into its TF-IDF vector
    # Result is a matrix (2D array): rows=job roles, columns=unique skills
    item_matrix = vectorizer.fit_transform(corpus)

    return vectorizer, item_matrix


# =============================================================================
# SECTION 5: THE RECOMMENDATION ENGINE — COSINE SIMILARITY
# =============================================================================
# Now we compare the USER PROFILE vector against every JOB ROLE vector.
#
# WHAT IS COSINE SIMILARITY?
#   - Measures the ANGLE between two vectors (not the distance)
#   - Why angle and not distance? Because we care about the DIRECTION of interests,
#     not how many skills someone listed. A user listing 2 matching skills and one
#     listing 10 matching skills should score similarly if the PATTERN is the same.
#
#   Formula: cos(θ) = (A · B) / (||A|| × ||B||)
#     - A · B = dot product (multiply matching elements, then sum)
#     - ||A|| = magnitude of vector A (square root of sum of squares)
#     - Result is between -1 and 1 (for TF-IDF: always 0 to 1)
#
#   Score Meaning:
#     1.0 → Perfect match (identical orientation)
#     0.5 → Moderate match (some overlap)
#     0.0 → No overlap at all
#
# WHY NOT EUCLIDEAN DISTANCE?
#   - Euclidean measures raw distance between points
#   - Problem: A job role description with 20 skills would ALWAYS seem farther away
#     than one with 5 skills, even if they match perfectly
#   - Cosine is MAGNITUDE-INVARIANT — it ignores how long the vector is

def get_recommendations(user_skills_raw: str, vectorizer, item_matrix, job_roles: list, top_n: int = 3) -> list:
    """
    THE FULL 4-STEP PIPELINE:

    Step 1 — Ingestion:  Normalize the user's raw skill input
    Step 2 — Scoring:    Transform user skills using the SAME vectorizer (same vocab!)
                         then calculate cosine similarity vs all job role vectors
    Step 3 — Sorting:    Sort scores highest to lowest
    Step 4 — Filtering:  Return only top_n results

    Returns a list of dicts: [{title, score, matched_skills}, ...]
    """

    # ── STEP 1: INGESTION ──────────────────────────────────────────────────
    # Normalize the user's raw input to fix abbreviations and casing
    user_skills_clean = normalize_skills(user_skills_raw)
    print(f"\n[SYSTEM] Normalized your input to: '{user_skills_clean}'")

    # ── STEP 2: SCORING ────────────────────────────────────────────────────
    # Transform user skills using the SAME vectorizer (critical!)
    # We use vectorizer.transform() — NOT fit_transform() — because the vocabulary
    # is already learned. We just convert the user's skills into the same vector space.
    # If the user types a skill not in our vocabulary, it gets ignored (score=0 for that dimension).
    user_vector = vectorizer.transform([user_skills_clean])

    # Calculate cosine similarity between the user vector and ALL item vectors at once
    # cosine_similarity() returns a 2D array. We take [0] to get the first (only) row.
    # Result: an array like [0.21, 0.87, 0.34, ...] — one score per job role
    scores = cosine_similarity(user_vector, item_matrix)[0]

    # ── STEP 3: SORTING ────────────────────────────────────────────────────
    # np.argsort() returns the INDICES that would sort the array in ascending order
    # [::-1] reverses it → descending order (highest score first)
    # Example: if scores = [0.21, 0.87, 0.34], argsort = [0,2,1], reversed = [1,2,0]
    ranked_indices = np.argsort(scores)[::-1]

    # ── STEP 4: FILTERING ──────────────────────────────────────────────────
    # Take only the top_n indices
    top_indices = ranked_indices[:top_n]

    # Build the results list
    results = []
    
    for idx in top_indices:
        role = job_roles[idx]
        score = scores[idx]

        # BONUS: Find which specific skills matched
        # We do this by checking which of the user's normalized skills appear in the role
        user_skill_list = set(user_skills_clean.lower().split())
        role_skill_list = set(role["skills"].lower().split())
        matched = user_skill_list.intersection(role_skill_list)

        results.append({
            "title": role["title"],
            "score": score,
            "match_percent": round(score * 100, 1),
            "matched_skills": sorted(matched) if matched else ["(semantic match via TF-IDF)"]
        })

    return results


# =============================================================================
# SECTION 6: COLD START HANDLER
# =============================================================================
# The "Cold Start Problem": What happens when a user gives us NO skills?
# If the user vector is all zeros, cosine similarity = 0 for everything.
# Solution: Show "trending" / popular roles instead (default recommendations).

def cold_start_recommendations(job_roles: list, top_n: int = 3) -> list:
    """
    Called when no user input is given.
    Returns globally popular roles (we simulate popularity with a preset ranking).
    In a real system, this would pull from interaction logs / trending data.
    """
    # Simulated popularity ranking (in real life: query a database for click rates)
    popular_roles = ["Full Stack Developer", "Data Scientist", "Backend Developer"]
    results = []
    for role in job_roles:
        if role["title"] in popular_roles:
            results.append({"title": role["title"], "score": None, "match_percent": "N/A", "matched_skills": []})
    return results[:top_n]


# =============================================================================
# SECTION 7: DISPLAY FUNCTIONS — MAKING OUTPUT BEAUTIFUL
# =============================================================================

def display_results(results: list, is_cold_start: bool = False):
    """Prints the recommendation results in a clean, readable format."""
    print("\n" + "="*60)
    if is_cold_start:
        print("  🔥 TRENDING ROLES (No skills provided)")
        print("  (These are the most popular roles on the platform)")
    else:
        print("  🎯 TOP CAREER PATH RECOMMENDATIONS")
    print("="*60)

    for i, result in enumerate(results, start=1):
        print(f"\n  #{i} — {result['title']}")
        if result['match_percent'] != "N/A":
            # Create a visual progress bar
            bar_length = 30
            filled = int((result['score'] * bar_length))
            bar = "█" * filled + "░" * (bar_length - filled)
            print(f"      Match Score: [{bar}] {result['match_percent']}%")
            print(f"      Matched Skills: {', '.join(result['matched_skills'])}")
        print()

    print("="*60)


# =============================================================================
# SECTION 8: MAIN PROGRAM — THE USER INTERFACE
# =============================================================================

def main():
    """
    Entry point of the program.
    Handles the user interaction loop.
    """
    print("\n" + "="*60)
    print("   🤖 TECH STACK RECOMMENDER — DecodeLabs Project 3")
    print("   Powered by TF-IDF + Cosine Similarity")
    print("="*60)
    print("\nThis system recommends the best tech career path for YOU")
    print("based on the skills you already have.\n")

    # BUILD THE MODEL ONCE (expensive — only do this once)
    # In production, you'd cache this and only rebuild when the dataset changes
    vectorizer, item_matrix = build_tfidf_model(JOB_ROLES)
    print(f"[SYSTEM] Model loaded. Vocabulary size: {len(vectorizer.vocabulary_)} unique skill terms.")
    print(f"[SYSTEM] Catalogue: {len(JOB_ROLES)} job roles available.\n")

    # THE MAIN INTERACTION LOOP
    while True:
        print("-"*60)
        print("Enter your skills (comma-separated).")
        print("Example: Python, SQL, Machine Learning, Statistics")
        print("Type 'quit' to exit, 'explore' to see all roles.\n")

        raw_input = input("Your skills: ").strip()

        # Exit condition
        if raw_input.lower() == 'quit':
            print("\n Thanks for using the Tech Stack Recommender!")
            break

        # Explore mode — show all roles
        if raw_input.lower() == 'explore':
            print("\n📋 ALL AVAILABLE JOB ROLES IN THE CATALOGUE:")
            for i, role in enumerate(JOB_ROLES, 1):
                print(f"  {i:>2}. {role['title']}")
                print(f"       Skills: {role['skills']}")
            print()
            continue

        # Cold start: user gave no input
        if not raw_input:
            print("\n⚠️  No skills entered. Showing trending roles instead...")
            results = cold_start_recommendations(JOB_ROLES)
            display_results(results, is_cold_start=True)
            continue

        # Validate: minimum 3 skills required (as per project spec)
        skills_count = len([s for s in raw_input.split(",") if s.strip()])
        if skills_count < 3:
            print(f"\n⚠️  You entered {skills_count} skill(s). Please enter at least 3 for accurate matching.")
            continue

        # Run the recommendation engine
        try:
            results = get_recommendations(
                user_skills_raw=raw_input,
                vectorizer=vectorizer,
                item_matrix=item_matrix,
                job_roles=JOB_ROLES,
                top_n=3  # Return Top 3 (change this number to get more/fewer results)
            )

            # Display results
            display_results(results)

        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again with valid skills.")


# =============================================================================
# SECTION 9: ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
