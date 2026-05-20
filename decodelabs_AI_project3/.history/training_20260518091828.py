"""
=============================================================================
PROJECT 3: AI RECOMMENDATION LOGIC — TECH STACK RECOMMENDER
DecodeLabs Industrial Training Kit | Batch 2026
=============================================================================
"""


import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
        "skills": "Python Java SQL APIs REST Databases Docker FastAPI Microservices Node Git"
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

def normalize_skills(raw_input : str) -> str :

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


def build_tfidf_model(job_roles : list) -> tuple :

    corpus = [role["skills"] for role in job_roles]

    vectorizer = TfidfVectorizer(
        analyzer='word',
        sublinear_tf=True  # log normalization for TF
    )

    item_matrix = vectorizer.fit_transform(corpus)

    return vectorizer , item_matrix


def get_recommendations(user_skills_raw: str, vectorizer, item_matrix, job_roles: list, top_n: int = 3) -> list:
    
    user_skills_clean = normalize_skills(user_skills_raw)
    print(f"\n[SYSTEM] Normalized your input to: '{user_skills_clean}'")
    
    user_vector = vectorizer.transform([user_skills_clean])
    
    scores = cosine_similarity(user_vector, item_matrix)[0]
    
    ranked_indices = np.argsort(scores)[::-1]
    
    top_indices = ranked_indices[:top_n]

    results = []

    for idx in top_indices:
        role = job_roles[idx]
        score = scores[idx]
        
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


def cold_start_recommendations(job_roles: list, top_n: int = 3) -> list :
     
    popular_roles = ["Full Stack Developer", "Data Scientist", "Backend Developer"]
    results = []
    for role in job_roles:
        if role["title"] in popular_roles:
            results.append({"title": role["title"], "score": None, "match_percent": "N/A", "matched_skills": []})
    return results[:top_n]



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

ef main():
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

            # EDGE CASE TEST PROMPT: push user to explore limits
            print("\n🔬 EDGE CASE IDEAS TO TEST:")
            print("  1. Type skills for TWO different careers (e.g. Python, React, Docker, Kubernetes)")
            print("  2. Type ALL skills — which role wins?")
            print("  3. Type only 1 skill (e.g. 'SQL') and see what happens")
            print("  4. Try typos: 'Pythn, SQl' — does normalization catch them?")
            print("  5. Try aliases: 'ML, DL, JS, K8s' — do they expand correctly?")

        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again with valid skills.")



if __name__ == "__main__":
    main()










