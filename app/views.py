from django.shortcuts import render, redirect, get_object_or_404
from sentence_transformers import SentenceTransformer, util
from .models import Job, Resume
from .ml_utils import (
    build_feature_vector,
    extract_years_experience,
    extract_education_level,
)
from django.db.models import Avg
from django.db.models import Q
from django.contrib.auth.decorators import login_required
from django.conf import settings
import PyPDF2
import re
import csv
import os

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False


# 🔥 ML SHORTLIST CLASSIFIER (learned weights instead of a fixed formula)
# Once enough real shortlist/reject decisions have been made and
# `python manage.py train_shortlist_model` has been run, this loads the
# trained scikit-learn model. Until then, SHORTLIST_MODEL stays None and
# upload_resume falls back to the original fixed-weight formula — so the
# app keeps working immediately even before any training has happened.
#
# 🔥 GRADUAL TRUST TIERS
# A model trained on only a handful of examples is unstable — the exact
# same resume can get a wildly different score every time it's retrained.
# Instead of fully trusting the model the moment ANY training has
# happened, trust is earned gradually based on how much labeled data and
# cross-validated accuracy it actually has:
#
#   < MIN_SAMPLES_FOR_ML samples      -> ignore ML entirely, formula only
#   accuracy below MIN_ACCURACY       -> ignore ML entirely, formula only
#   samples below FULL_TRUST_SAMPLES  -> 50/50 blend of formula + ML
#   samples >= FULL_TRUST_SAMPLES     -> full ML prediction
#
# This is what stops the "same resume, completely different result"
# problem — a shaky model can no longer swing the score on its own.

MIN_SAMPLES_FOR_ML = 15       # below this, don't trust the model at all
FULL_TRUST_SAMPLES = 30       # above this, trust the model fully
MIN_ACCURACY = 0.65           # below this cross-validated accuracy, ignore it

SHORTLIST_MODEL_PATH = os.path.join(settings.BASE_DIR, 'app', 'ml_model', 'shortlist_classifier.pkl')
SHORTLIST_MODEL_META_PATH = os.path.join(settings.BASE_DIR, 'app', 'ml_model', 'shortlist_classifier_meta.json')

SHORTLIST_MODEL = None
SHORTLIST_MODEL_META = None

if HAS_JOBLIB and os.path.exists(SHORTLIST_MODEL_PATH):
    try:
        SHORTLIST_MODEL = joblib.load(SHORTLIST_MODEL_PATH)
    except Exception:
        SHORTLIST_MODEL = None

if os.path.exists(SHORTLIST_MODEL_META_PATH):
    try:
        import json
        with open(SHORTLIST_MODEL_META_PATH) as f:
            SHORTLIST_MODEL_META = json.load(f)
    except Exception:
        SHORTLIST_MODEL_META = None


def get_ml_trust_level():
    """
    Returns 'none', 'blend', or 'full' based on how much labeled data and
    accuracy the trained model actually has behind it.
    """
    if SHORTLIST_MODEL is None or SHORTLIST_MODEL_META is None:
        return 'none'

    n_samples = SHORTLIST_MODEL_META.get('n_samples', 0)
    cv_accuracy = SHORTLIST_MODEL_META.get('cv_accuracy')

    if n_samples < MIN_SAMPLES_FOR_ML:
        return 'none'
    if cv_accuracy is not None and cv_accuracy < MIN_ACCURACY:
        return 'none'
    if n_samples < FULL_TRUST_SAMPLES:
        return 'blend'
    return 'full'


# 🔥 BETTER TEXT EXTRACTION
# PyPDF2 frequently mangles multi-column resumes and icon-based contact
# sections — merging lines together, dropping spaces, or losing symbols
# like "@". pdfplumber preserves layout/spacing far more reliably, so we
# try it first and only fall back to PyPDF2 if it's unavailable or fails.
def extract_pdf_text(file):
    text = ""

    if HAS_PDFPLUMBER:
        try:
            file.seek(0)
            with pdfplumber.open(file) as pdf_doc:
                for page in pdf_doc.pages:
                    text += (page.extract_text() or "") + "\n"
        except Exception:
            text = ""

    if not text.strip():
        try:
            file.seek(0)
            pdf_doc = PyPDF2.PdfReader(file)
            for page in pdf_doc.pages:
                text += page.extract_text() or ""
        except Exception:
            text = ""

    return text


# 🔥 EMAIL EXTRACTION WITH FALLBACK
# Many resume templates put the email in a sidebar with an icon, as a
# clickable hyperlink rather than plain visible text. PyPDF2 sometimes
# fails to extract the "@" / "." from these because of font/encoding
# quirks, or the email exists only as a link annotation with no matching
# text at all. This checks BOTH the extracted text AND the PDF's
# embedded hyperlink annotations (mailto: links) before giving up.

EMAIL_REGEX = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'


def extract_email(pdf_reader, raw_text):
    match = re.search(EMAIL_REGEX, raw_text)
    if match:
        return match.group()

    # Fallback: look for mailto: links embedded in the PDF itself
    try:
        if pdf_reader is not None:
            for page in pdf_reader.pages:
                annots = page.get("/Annots")
                if not annots:
                    continue
                for annot_ref in annots:
                    try:
                        annot = annot_ref.get_object()
                        uri = annot.get("/A", {}).get("/URI", "")
                        if uri and uri.lower().startswith("mailto:"):
                            candidate = uri[7:].strip()
                            if re.match(EMAIL_REGEX, candidate):
                                return candidate
                    except Exception:
                        continue
    except Exception:
        pass

    # Fallback 2: PyPDF2 sometimes drops the "@" and "." characters
    # entirely due to font/encoding quirks, turning "name@gmail.com"
    # into "name gmail com" with no symbols left at all. Reconstruct it
    # by looking for a username immediately followed by a known
    # email provider name.
    known_domains = ["gmail", "yahoo", "outlook", "hotmail", "rediffmail", "icloud", "protonmail"]
    domain_pattern = "|".join(known_domains)
    loose_match = re.search(
        r'([a-zA-Z0-9._-]{3,30})\s+(' + domain_pattern + r')\s*[. ]?\s*(com|in|co\.in)\b',
        raw_text, re.IGNORECASE
    )
    if loose_match:
        username, domain, tld = loose_match.groups()
        tld = tld.replace(".", "")
        return f"{username}@{domain}.{tld}"

    return "Not found"


# 🔥 LOAD MODEL ONCE
model = SentenceTransformer('all-mpnet-base-v2')
# 🔥 IMPROVED SKILL MATCHING
# Old logic: `skill in text_clean` — a naive substring check that caused
# real false positives ("c" matches inside "science"/"college", "java"
# matches inside "javascript") and false negatives (no handling of
# common variants like "ML" vs "machine learning", "ReactJS" vs "react").

SKILL_VARIANTS = {
    "javascript": ["javascript", "js"],
    "react": ["react", "reactjs", "react js"],
    "node": ["node", "nodejs", "node js"],
    "python": ["python", "py"],
    "machine learning": ["machine learning", "ml"],
    "deep learning": ["deep learning", "dl"],
    "nlp": ["nlp", "natural language processing"],
    "computer vision": ["computer vision", "cv"],
    "sql": ["sql", "mysql", "postgresql", "sqlite", "mssql", "database"],
    "html": ["html", "html5"],
    "css": ["css", "css3"],
    "c++": ["c++", "cpp"],
    "c#": ["c#", "csharp", "c sharp"],
    "power bi": ["power bi", "powerbi"],
    "data analysis": ["data analysis", "data analytics"],
    "data science": ["data science", "data scientist"],
}


def _skill_variant_found(variant, text):
    """Word-boundary-safe match. Uses lookaround instead of \\b because
    \\b doesn't handle symbols like '+' or '#' in skills like c++/c#."""
    pattern = r'(?<![a-zA-Z0-9])' + re.escape(variant) + r'(?![a-zA-Z0-9])'
    return re.search(pattern, text) is not None


def get_matched_skills(job_skills, text_clean, skills_section_text=""):
    """
    Returns which of the job's required skills are present in the resume.

    Pass 1 — exact/alias keyword matching (word-boundary safe, so "c"
    won't match inside "science", and common abbreviations like
    js/ml/dl/cv are recognized as the same skill).

    Pass 2 — semantic fallback using sentence embeddings, for skills that
    aren't named literally but are clearly described (e.g. resume says
    "trained predictive models" instead of the words "machine learning").
    """
    matched = []
    unmatched = []

    for skill in job_skills:
        skill_clean = skill.strip().lower()
        if not skill_clean:
            continue

        variants = SKILL_VARIANTS.get(skill_clean, [skill_clean])
        if any(_skill_variant_found(v, text_clean) for v in variants):
            matched.append(skill_clean)
        else:
            unmatched.append(skill_clean)

    if unmatched and skills_section_text.strip():
        try:
            skill_embs = model.encode(unmatched, convert_to_tensor=True)
            section_emb = model.encode(skills_section_text, convert_to_tensor=True)
            sims = util.cos_sim(skill_embs, section_emb)
            for i, skill in enumerate(unmatched):
                if sims[i][0].item() >= 0.45:
                    matched.append(skill)
        except Exception:
            pass

    return matched


# 🔥 IMPROVED SECTION EXTRACTION
# Old version just grabbed everything after the FIRST occurrence of a
# keyword, with no stopping point — so "Experience" ended up containing
# the entire rest of the resume (education, skills, projects, everything).
#
# This version finds ALL section headers in the resume, sorts them by
# position, and cuts each section off at wherever the NEXT header starts.
# That keeps sections clean and separated from each other.

SECTION_HEADER_MAP = {
    "skills": "skills",
    "technical skills": "skills",
    "soft skills": "skills",
    "core competencies": "skills",

    "experience": "experience",
    "work experience": "experience",
    "professional experience": "experience",
    "internship": "experience",
    "internships": "experience",
    "employment history": "experience",

    "education": "education",
    "academic background": "education",
    "qualification": "education",
    "qualifications": "education",

    # Not scored directly, but still act as boundaries so they don't
    # bleed into skills/experience/education
    "projects": None,
    "certifications": None,
    "certification": None,
    "achievements": None,
    "objective": None,
    "summary": None,
    "profile": None,
    "career objective": None,
    "declaration": None,
    "hobbies": None,
    "languages known": None,
    "references": None,
}


def extract_sections(text, max_len=600):
    text = text.lower()

    # Find every header occurrence, longest phrases first so
    # "work experience" matches before the bare "experience" inside it
    headers_sorted = sorted(SECTION_HEADER_MAP.keys(), key=len, reverse=True)

    positions = []
    covered = set()
    for header in headers_sorted:
        for m in re.finditer(r'\b' + re.escape(header) + r'\b', text):
            # avoid re-matching a shorter header inside an already-matched longer one
            span = set(range(m.start(), m.end()))
            if span & covered:
                continue
            covered |= span
            positions.append((m.start(), m.end(), header))

    positions.sort(key=lambda p: p[0])

    sections = {"skills": "", "experience": "", "education": ""}

    for i, (start, end, header) in enumerate(positions):
        category = SECTION_HEADER_MAP.get(header)
        if not category:
            continue

        next_start = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        content = text[end:next_start].strip()

        # keep the richest match if a category header appears more than once
        if len(content) > len(sections[category]):
            sections[category] = content[:max_len]

    return sections["skills"], sections["experience"], sections["education"]
# 🟢 HOME PAGE
@login_required
def home(request):
    jobs = Job.objects.all().order_by('-id')

    job_data = []
    for job in jobs:
        resumes = Resume.objects.filter(job=job).order_by('-score')
        job_data.append({
            'job': job,
            'resumes': resumes
        })

    return render(request, 'app/home.html', {
        'job_data': job_data,
        'total_jobs': Job.objects.count(),
        'total_resumes': Resume.objects.count(),
        'avg_score': round(Resume.objects.aggregate(avg=Avg('score'))['avg'] or 0, 2),
        'shortlisted': Resume.objects.filter(score__gte=70).count()
    })


# 🟢 JOB LIST PAGE
def jobs(request):
    jobs = Job.objects.all().order_by('-id')

    # 🔥 Search bar now actually works — filters by title or description
    query = request.GET.get('q')
    if query:
        jobs = jobs.filter(
            Q(title__icontains=query) | Q(description__icontains=query)
        )

    for job in jobs:
        job.skill_list = [s.strip() for s in (job.skills or "").split(',') if s.strip()]

    return render(request, 'app/jobs.html', {'jobs': jobs})


# 🟢 CREATE JOB
def create_job(request):
    if request.method == 'POST':

        def parse_float(value):
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        Job.objects.create(
            title=request.POST.get('title'),
            description=request.POST.get('description'),
            skills=request.POST.get('skills'),
            experience=request.POST.get('experience'),
            education=request.POST.get('education'),
            location=request.POST.get('location'),

            qualification=request.POST.get('qualification'),
            eligible_branches=request.POST.get('eligible_branches'),
            current_semester=request.POST.get('current_semester'),
            expected_graduation=request.POST.get('expected_graduation'),

            min_tenth_percentage=parse_float(request.POST.get('min_tenth_percentage')),
            min_twelfth_percentage=parse_float(request.POST.get('min_twelfth_percentage')),
            min_cgpa=parse_float(request.POST.get('min_cgpa')),

            stipend=request.POST.get('stipend'),
            work_mode=request.POST.get('work_mode'),
            additional_requirements=request.POST.get('additional_requirements'),
        )
        return redirect('/jobs/')

    return render(request, 'app/create_job.html')


# 🟢 UPLOAD RESUME
@login_required
def upload_resume(request, job_id):
    job = get_object_or_404(Job, id=job_id)

    import re
    import PyPDF2

    # =========================
    # 👤 NAME EXTRACTION
    # =========================
    # 🔥 Section headings that can look like a "clean short name line"
    # but are NOT names — e.g. "Area of Interest" (3 words) used to beat
    # the real name "Tharun K" (2 words) under the old "pick longest"
    # rule, since it just chose whichever candidate had the most words.
    SECTION_HEADING_BLOCKLIST = {
        "area of interest", "areas of interest", "career objective",
        "technical skills", "personal details", "personal information",
        "declaration", "hobbies", "hobbies and interests", "interests",
        "extracurricular activities", "co curricular activities",
        "achievements", "certifications", "certification", "projects",
        "key skills", "soft skills", "strengths", "languages known",
        "professional summary", "summary", "profile", "about me",
        "work experience", "professional experience", "internships",
        "academic details", "academic background", "skills summary"
    }

    def extract_name(text):
        lines = text.split('\n')
        candidates = []

        for line in lines[:10]:
            line = line.strip()

            if not line:
                continue

            if len(line) < 3 or len(line) > 40:
                continue
            if "@" in line:
                continue
            if re.search(r'\d', line):
                continue
            if any(word in line.lower() for word in [
                "resume", "curriculum", "vitae",
                "email", "phone", "contact",
                "linkedin", "github", "objective"
            ]):
                continue
            if line.lower().strip() in SECTION_HEADING_BLOCKLIST:
                continue

            if re.match(r'^[A-Za-z\s]+$', line):
                candidates.append(line)

        if candidates:
            # 🔥 FIXED: names sit at the very top of a resume almost
            # always — picking the FIRST valid candidate is far more
            # reliable than picking whichever one has the most words
            # (which wrongly favored 3-word section headings over a
            # genuine 2-word name).
            return candidates[0].title()

        # 🔥 FALLBACK: the line-based approach fails on resumes where the
        # PDF has no real line breaks near the top (common with multi-column
        # layouts, where the name/contact block gets merged into one long
        # run of text). Instead, take the first few alphabetic words at the
        # very start of the document, stopping once we hit a junk word like
        # "email"/"gmail"/"resume" etc.
        JUNK_WORDS = {
            "resume", "curriculum", "vitae", "email", "phone", "contact",
            "linkedin", "github", "objective", "gmail", "yahoo", "outlook",
            "hotmail", "com", "profile", "summary", "cv", "address"
        }

        words = re.findall(r"[A-Za-z]+", text[:200])
        name_words = []
        for w in words:
            if w.lower() in JUNK_WORDS:
                if name_words:
                    break
                continue
            name_words.append(w)
            if len(name_words) >= 3:
                break

        if name_words:
            return " ".join(name_words).title()

        return "Unknown"

    if request.method == 'POST':
        files = request.FILES.getlist('files')

        for file in files:

            # =========================
            # 📄 EXTRACT TEXT (pdfplumber first, PyPDF2 fallback)
            # =========================
            text = extract_pdf_text(file)

            # =========================
            # 🧹 CLEAN TEXT (STEP 3)
            # =========================
            text_clean = text.lower()
            skills_text, exp_text, edu_text = extract_sections(text_clean)
            text_clean = re.sub(r'\S+@\S+', ' ', text_clean)          # remove email
            text_clean = re.sub(r'http\S+|www\S+', ' ', text_clean)   # remove links
            text_clean = re.sub(r'\+?\d[\d\s\-]{8,}', ' ', text_clean) # remove phone
            text_clean = re.sub(r'[^a-z\s]', ' ', text_clean)         # remove symbols
            text_clean = re.sub(r'\s+', ' ', text_clean).strip()      # remove extra spaces

            # =========================
            # 📧 EMAIL (with hyperlink fallback)
            # =========================
            # NOTE: need a fresh PdfReader since `file` stream was
            # consumed above; re-open it for the annotation check
            file.seek(0)
            try:
                pdf_for_email = PyPDF2.PdfReader(file)
            except Exception:
                pdf_for_email = None
            email = extract_email(pdf_for_email, text)

            # =========================
            # 📱 PHONE
            # =========================
            phone_match = re.search(r'(\+91[\-\s]?|0)?[6-9]\d{9}', text)
            phone = phone_match.group() if phone_match else "Not found"

            # =========================
            # 👤 NAME
            # =========================
            name = extract_name(text)

            # =========================
            # 🤖 STEP 2: JOB CONTEXT
            # =========================
            job_text = f"""
            Job Title: {job.title}
            Required Skills: {job.skills}
            Experience Required: {job.experience}
            Education Required: {job.education}
            Job Description: {job.description}
            """

        

            # =========================
            # 🔧 RULE BASED
            # =========================

            # SKILLS
            job_skills = [s.strip() for s in (job.skills or "").lower().split(',') if s.strip()]
            matched_skills = get_matched_skills(job_skills, text_clean, skills_text)

            skills_score = int((len(matched_skills) / len(job_skills)) * 100) if job_skills else 0

            # EXPERIENCE
            if any(word in text_clean for word in ["experience", "intern", "worked", "project"]):
                experience_score = 70
            else:
                experience_score = 30

            # EDUCATION
            if any(word in text_clean for word in ["btech", "degree", "college", "university"]):
                education_score = 70
            else:
                education_score = 30
            # =========================
# 🤖 SECTION-BASED AI MATCHING
# =========================

# JOB PARTS
            job_skills_text = f"Skills: {job.skills}"
            job_exp_text = f"Experience: {job.experience}"
            job_edu_text = f"Education: {job.education}"

# RESUME PARTS → EMBEDDING
            skills_emb = model.encode(skills_text, convert_to_tensor=True)
            exp_emb = model.encode(exp_text, convert_to_tensor=True)
            edu_emb = model.encode(edu_text, convert_to_tensor=True)

# JOB EMBEDDINGS
            job_skills_emb = model.encode(job_skills_text, convert_to_tensor=True)
            job_exp_emb = model.encode(job_exp_text, convert_to_tensor=True)
            job_edu_emb = model.encode(job_edu_text, convert_to_tensor=True)

# SIMILARITY
            skills_sim = util.cos_sim(skills_emb, job_skills_emb).item()
            exp_sim = util.cos_sim(exp_emb, job_exp_emb).item()
            edu_sim = util.cos_sim(edu_emb, job_edu_emb).item()

# CONVERT TO %
            skills_ai = max(0, int(skills_sim * 100))
            exp_ai = max(0, int(exp_sim * 100))
            edu_ai = max(0, int(edu_sim * 100))

            # =========================
            # 🔥 RICHER FEATURES FOR THE ML MODEL
            # =========================
            years_experience = extract_years_experience(text_clean)
            education_level = extract_education_level(text_clean)
            resume_word_count = len(text_clean.split())

            feature_vector = build_feature_vector(
                job=job,
                skills_ai=skills_ai,
                exp_ai=exp_ai,
                edu_ai=edu_ai,
                skill_match_ratio=skills_score,
                matched_skills_count=len(matched_skills),
                years_experience=years_experience,
                education_level=education_level,
                resume_word_count=resume_word_count
            )

            # =========================
            # 🎯 FINAL SCORE
            # =========================
            # 🔥 Trust in the ML model is now EARNED gradually based on
            # how much labeled data and accuracy it actually has — see
            # get_ml_trust_level(). This is what stops the same resume
            # from getting wildly different scores across retrains when
            # there was only a handful of shaky examples behind the model.
            formula_score = int((skills_ai * 0.5) + (exp_ai * 0.3) + (edu_ai * 0.2))
            trust_level = get_ml_trust_level()

            if trust_level == 'none':
                final_score = formula_score
                scoring_method = "formula"
            else:
                try:
                    proba = SHORTLIST_MODEL.predict_proba([feature_vector])[0][1]
                    ml_score = int(proba * 100)

                    if trust_level == 'full':
                        final_score = ml_score
                        scoring_method = "ml_model"
                    else:  # 'blend'
                        final_score = int((formula_score * 0.5) + (ml_score * 0.5))
                        scoring_method = "ml_blend"
                except Exception:
                    final_score = formula_score
                    scoring_method = "formula"
            # =========================
# 🤖 AI INSIGHTS
# =========================
            match_reasons = []

# AI strength
            if skills_ai >= 80:
                match_reasons.append("Strong match with job description")

# Skills
            if len(matched_skills) >= 3:
                match_reasons.append("Good skill match with required technologies")

# Experience (🔥 FIXED: was nested inside the skills check before,
# so a strong-experience candidate with weak skill overlap never got credit)
            if experience_score == 70:
                match_reasons.append("Has relevant experience or projects")

# Education
            if education_score == 70:
                match_reasons.append("Meets educational requirements")

# Fallback
            if not match_reasons:
                match_reasons.append("Basic profile match")

# Scoring method transparency
            if scoring_method == "ml_model":
                match_reasons.append("Score predicted by trained ML model")
            elif scoring_method == "ml_blend":
                match_reasons.append("Score blended: formula + early-stage ML model")
            # =========================
            # 💾 SAVE
            # =========================
            # 🔥 NOW SAVED: match_reasons + the exact score breakdown,
            # so resume_detail can just display these instead of
            # recomputing with different logic and showing a different number
            # Rewind again — the stream was read twice above (once for
            # text extraction, once for the email fallback check), and
            # Django needs to read it fresh from the start to save it to disk
            file.seek(0)

            resume = Resume.objects.create(
                job=job,
                name=name,
                email=email,
                phone=phone,
                file=file,
                extracted_text=text_clean,
                score=final_score,
                match_reasons=", ".join(match_reasons),
                matched_skills=", ".join(matched_skills),
                skills_score=skills_ai,
                experience_score=exp_ai,
                education_score=edu_ai,
                experience_text=(exp_text.strip().capitalize() if exp_text.strip() else "No experience section found"),
                education_text=(edu_text.strip().capitalize() if edu_text.strip() else "No education section found"),
                skill_match_ratio=skills_score,
                matched_skills_count=len(matched_skills),
                years_experience=years_experience,
                education_level=education_level,
                resume_word_count=resume_word_count
            )

        return redirect('/job/' + str(job.id) + '/')

    return render(request, 'app/upload_resume.html', {'job': job})


# 🟢 JOB DETAIL PAGE
@login_required
def job_detail(request, id):
    job = get_object_or_404(Job, id=id)

    skill_list = [s.strip() for s in (job.skills or "").split(',') if s.strip()]

    resumes = Resume.objects.filter(job=job)

    # ✅ SEARCH
    query = request.GET.get('q')
    if query:
        resumes = resumes.filter(name__icontains=query)

    # ✅ SORT
    sort = request.GET.get('sort')
    if sort == "high":
        resumes = resumes.order_by('-score')
    elif sort == "low":
        resumes = resumes.order_by('score')
    else:
        resumes = resumes.order_by('-score')  # default

    top_candidate = resumes.first() if resumes.exists() else None

    return render(request, 'app/job_detail.html', {
        'job': job,
        'resumes': resumes,
        'top_candidate': top_candidate,
        'skill_list': skill_list
    })


# 🟢 RESUME DETAIL PAGE
# 🔥 FIXED: this used to run a completely separate, cruder scoring
# algorithm than upload_resume, which is why the detail page could show
# a different score (e.g. 25%) than the list/ranking page (e.g. 69%)
# for the exact same candidate. Now it simply displays what was already
# computed and stored at upload time — one score, everywhere, always.
def resume_detail(request, id):
    resume = get_object_or_404(Resume, id=id)
    job = resume.job

    skill_list = [s.strip() for s in (job.skills or "").split(',') if s.strip()]

    matched_skills = [s.strip() for s in (resume.matched_skills or "").split(',') if s.strip()]
    match_reasons = [r.strip() for r in (resume.match_reasons or "").split(',') if r.strip()]

    job_skills = [s.strip() for s in (job.skills or "").lower().split(',') if s.strip()]
    skills_percent = int((len(matched_skills) / len(job_skills)) * 100) if job_skills else 0

    experience_percent = int(resume.experience_score or 0)
    education_percent = int(resume.education_score or 0)
    overall = int(resume.score or 0)

    return render(request, 'app/resume_detail.html', {
        'resume': resume,
        'matched_skills': matched_skills,
        'skills_percent': skills_percent,
        'experience_percent': experience_percent,
        'education_percent': education_percent,
        'overall': overall,
        'skill_list': skill_list,
        'match_reasons': match_reasons,
        'experience_text': resume.experience_text,
        'education_text': resume.education_text
    })

# 🟢 ANALYTICS PAGE
@login_required
def analytics(request):
    from django.db.models import Avg, Count
    from .models import Resume, Job

    resumes = Resume.objects.all()

    # ✅ FIXED RANGES
    ranges = {
        "r1": 0,  # 90-100
        "r2": 0,  # 80-89
        "r3": 0,  # 70-79
        "r4": 0,  # 60-69
        "r5": 0   # below 60
    }

    for r in resumes:
        if r.score is None:
            continue
        elif r.score >= 90:
            ranges["r1"] += 1
        elif r.score >= 80:
            ranges["r2"] += 1
        elif r.score >= 70:
            ranges["r3"] += 1
        elif r.score >= 60:
            ranges["r4"] += 1
        else:
            ranges["r5"] += 1

    # ✅ JOB STATS
    job_stats = Job.objects.annotate(
        total_resumes=Count('resume'),
        avg_score=Avg('resume__score')
    )

    return render(request, 'app/analytics.html', {
        'ranges': ranges,
        'job_stats': job_stats
    })

def download_csv(request, job_id):
    job = get_object_or_404(Job, id=job_id)
    resumes = Resume.objects.filter(job=job)

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="{job.title}_candidates.csv"'

    writer = csv.writer(response)
    writer.writerow(['Name', 'Email', 'Phone', 'Score'])

    for r in resumes:
        writer.writerow([r.name, r.email, r.phone, r.score])

    return response


# 🟢 RECORD REAL SHORTLIST/REJECT DECISION
# 🔥 This is what actually feeds the ML classifier. Every time a
# recruiter clicks Shortlist or Reject here, it becomes one labeled
# training example. `python manage.py train_shortlist_model` uses these
# to learn real weights instead of the fixed 0.5/0.3/0.2 formula.
@login_required
def mark_shortlist(request, id):
    resume = get_object_or_404(Resume, id=id)

    if request.method == 'POST':
        decision = request.POST.get('decision')
        if decision == 'yes':
            resume.shortlisted = True
        elif decision == 'no':
            resume.shortlisted = False
        resume.save()

    return redirect('resume_detail', id=id)