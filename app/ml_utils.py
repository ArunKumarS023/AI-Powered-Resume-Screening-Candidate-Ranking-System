"""
Shared feature-engineering logic for the shortlist ML model.

CRITICAL: both `train_shortlist_model` (training) and `upload_resume`
(prediction) import `build_feature_vector` from HERE, and only here.
If training and prediction ever build feature vectors differently, the
model's predictions become meaningless (a classic "train/serve skew"
bug) — so there must be exactly one place this logic lives.
"""

import re


EDUCATION_LEVELS = {
    "phd": 4, "doctorate": 4,
    "master": 3, "m.tech": 3, "mtech": 3, "mba": 3, "msc": 3, "m.sc": 3,
    "post graduate": 3, "postgraduate": 3,
    "bachelor": 2, "b.tech": 2, "btech": 2, "b.e": 2, "bsc": 2, "b.sc": 2, "bca": 2,
    "under graduate": 2, "undergraduate": 2,
    "diploma": 1, "hsc": 1, "higher secondary": 1, "12th": 1,
}

FEATURE_NAMES = [
    "skills_ai", "experience_ai", "education_ai",
    "skill_match_ratio", "matched_skills_count",
    "years_experience", "education_level",
    "meets_experience_requirement", "meets_education_requirement",
    "resume_word_count",
]


def extract_years_experience(text):
    """Finds patterns like '3 years', '2+ yrs' and returns the largest
    number mentioned as a best-guess total years of experience."""
    text = (text or "").lower()
    matches = re.findall(r'(\d+)\+?\s*(?:years|yrs|year)\b', text)
    numbers = [int(m) for m in matches if m.isdigit()]
    return max(numbers) if numbers else 0


def extract_education_level(text):
    """Returns the highest education level mentioned in resume text as
    an ordinal: 0=none detected, 1=diploma/12th, 2=bachelor's, 3=master's, 4=phd."""
    text = (text or "").lower()
    level = 0
    for keyword, value in EDUCATION_LEVELS.items():
        if keyword in text and value > level:
            level = value
    return level


def parse_required_experience(exp_str):
    """Parses job.experience strings like '0-3', '2+', '2+ years' into
    a minimum required years number."""
    if not exp_str:
        return 0
    match = re.search(r'(\d+)', exp_str)
    return int(match.group(1)) if match else 0


def parse_required_education(edu_str):
    """Maps a job's free-text education requirement to the same ordinal
    scale as extract_education_level."""
    if not edu_str:
        return 0
    edu_str = edu_str.lower()
    level = 0
    for keyword, value in EDUCATION_LEVELS.items():
        if keyword in edu_str and value > level:
            level = value
    if "pg" in edu_str or "post grad" in edu_str:
        level = max(level, 3)
    if "ug" in edu_str or "under grad" in edu_str or "graduate" in edu_str:
        level = max(level, 2)
    return level


def build_feature_vector(job, skills_ai, exp_ai, edu_ai, skill_match_ratio,
                          matched_skills_count, years_experience,
                          education_level, resume_word_count):
    """
    Builds the exact feature vector the ML model trains on and predicts
    from. Always call this the same way at both training time and
    prediction time — never construct the feature list by hand elsewhere.
    """
    required_years = parse_required_experience(job.experience)
    required_edu_level = parse_required_education(job.education)

    meets_experience = 1 if years_experience >= required_years else 0
    meets_education = 1 if education_level >= required_edu_level else 0

    return [
        skills_ai,
        exp_ai,
        edu_ai,
        skill_match_ratio,
        matched_skills_count,
        years_experience,
        education_level,
        meets_experience,
        meets_education,
        resume_word_count,
    ]