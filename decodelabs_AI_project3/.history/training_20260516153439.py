"""
=============================================================================
PROJECT 3: AI RECOMMENDATION LOGIC — TECH STACK RECOMMENDER
DecodeLabs Industrial Training Kit | Batch 2026
=============================================================================
"""


import numpy as np
from sklearn.feature_extraction import TfidfVectorizer
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
        lower_skill = skill.lower()
        if lower_skill in SKILL_ALIASES :
            normalized.append(SKILL_ALIASES[lower_skill])
        else:
            normalized.append(skill.title())

    return " ".join(normalized)        


def build_tfidf_model(job_roles : list) -> tuple :
    corpus = [role["skills"] for role in job_roles]

    vectorizer = TfidfVectorizer(
        analyzer = "word" ,
        sublinear_tf = True
    )

    item_matrix = vectorizer.fit_transform(corpus)

    return vectorizer , item_matrix


def get_recommendations(user_skills_raw: str, vectorizer, item_matrix, job_roles: list, top_n: int = 3) -> list:
    
    user_skills_clean = normalize_skills(user_skills_raw)
    print(f"\n[SYSTEM] Normalized your input to: '{user_skills_clean}'")

    user_vector = vectorizer.transform([user_skills_clean])

    scores = cosine_similarity(user_vector , item_matrix)[0]

    ranked_indices = np.argsort(scores)[::-1]

    top_indices = ranked_indices[:top_n]

    results = []

    for idx in top_indices:
        role = job_roles[idx]
        score = scores[idx]
        
        user_skill_list = set(user_skills_clean.split().lower())
        role_skill_list = set(role["sk"])











