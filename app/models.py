from django.db import models

# Job Model
class Job(models.Model):
    title = models.CharField(max_length=200)
    description = models.TextField()

    skills = models.CharField(max_length=300, blank=True)
    experience = models.CharField(max_length=100, blank=True)
    education = models.CharField(max_length=200, blank=True)
    location = models.CharField(max_length=100, blank=True)

    # 🔥 NEW: Eligibility criteria (Academic Criteria + Student Eligibility
    # + Internship details, matching the internship posting format)
    qualification = models.CharField(max_length=100, blank=True)         # e.g. "B.E. / B.Tech"
    eligible_branches = models.CharField(max_length=300, blank=True)      # e.g. "CSE, IT, AIML, AIDS"
    current_semester = models.CharField(max_length=50, blank=True)        # e.g. "7th semester"
    expected_graduation = models.CharField(max_length=50, blank=True)     # e.g. "June/July 2027"

    min_tenth_percentage = models.FloatField(null=True, blank=True)       # e.g. 75.0
    min_twelfth_percentage = models.FloatField(null=True, blank=True)     # e.g. 70.0
    min_cgpa = models.FloatField(null=True, blank=True)                   # e.g. 6.5

    stipend = models.CharField(max_length=100, blank=True)                # e.g. "₹25,000/month"
    work_mode = models.CharField(max_length=50, blank=True)               # e.g. "Work from Office"
    additional_requirements = models.TextField(blank=True)                # relocation, full-time commitment etc.

    def __str__(self):
        return self.title


# Resume Model
class Resume(models.Model):
    job = models.ForeignKey(Job, on_delete=models.CASCADE)
    file = models.FileField(upload_to='resumes/')
    name = models.CharField(max_length=200)
    email = models.EmailField()
    phone = models.CharField(max_length=20)
    extracted_text = models.TextField(blank=True)
    score = models.FloatField(null=True, blank=True)

    # 🔥 NEW: store the exact breakdown computed at upload time
    # so resume_detail never has to recompute (and disagree) with the list score
    match_reasons = models.TextField(blank=True, null=True)
    matched_skills = models.TextField(blank=True, null=True)
    skills_score = models.FloatField(null=True, blank=True)
    experience_score = models.FloatField(null=True, blank=True)
    education_score = models.FloatField(null=True, blank=True)

    # 🔥 NEW: the actual extracted section snippets, so the detail page
    # can show *what was found* alongside the score, instead of just a number
    experience_text = models.TextField(blank=True, null=True)
    education_text = models.TextField(blank=True, null=True)

    # 🔥 NEW: the real human decision on this candidate.
    # None = no decision made yet, True = shortlisted, False = rejected.
    # This is the ground-truth label the ML classifier learns from —
    # without this, there's nothing to train a real model on.
    shortlisted = models.BooleanField(null=True, blank=True, default=None)

    # 🔥 NEW: richer features for the ML model, so it has more than just
    # 3 numbers to learn from
    skill_match_ratio = models.FloatField(null=True, blank=True)
    matched_skills_count = models.IntegerField(null=True, blank=True)
    years_experience = models.IntegerField(null=True, blank=True)
    education_level = models.IntegerField(null=True, blank=True)
    resume_word_count = models.IntegerField(null=True, blank=True)

    def __str__(self):
        return self.name