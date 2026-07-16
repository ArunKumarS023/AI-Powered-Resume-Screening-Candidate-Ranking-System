"""
Trains a scikit-learn classifier to predict shortlist decisions,
replacing the fixed 0.5/0.3/0.2 formula with weights learned from real
recruiter decisions.

🔥 NOW WITH REAL EVALUATION:
Instead of just training and trusting it blindly, this compares two
model types (Logistic Regression vs Random Forest) using cross-validation
— meaning each model is tested on resumes it never saw during that fold's
training — and picks whichever actually performs better, then reports
precision/recall/F1/confusion matrix so you can judge the result yourself
instead of taking it on faith.

🔥 NOW WITH GRADUAL TRUST:
Saves a metadata file alongside the model recording how many examples it
was trained on and its cross-validated accuracy. views.py uses this to
decide how much to trust the model — full trust only once there's enough
data, blended with the formula in between, ignored entirely below a
minimum. This is what stops scores from swinging wildly between retrains
when there are still only a handful of labeled examples.

Usage:
    python manage.py train_shortlist_model

Requirements:
    At least a handful of resumes must have been marked Shortlisted or
    Rejected (via the button on the resume detail page) before this can
    train anything meaningful. The more decisions recorded, the better
    the model gets — this is designed to be re-run periodically as more
    decisions accumulate.
"""

import os
import json
import joblib
import numpy as np

from django.core.management.base import BaseCommand
from django.conf import settings
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix

from app.models import Resume
from app.ml_utils import build_feature_vector

MODEL_DIR = os.path.join(settings.BASE_DIR, 'app', 'ml_model')
MODEL_PATH = os.path.join(MODEL_DIR, 'shortlist_classifier.pkl')
MODEL_META_PATH = os.path.join(MODEL_DIR, 'shortlist_classifier_meta.json')

MIN_TO_TRAIN = 4          # absolute floor — can't train with fewer than this
RECOMMENDED_MIN = 15      # a small but more trustworthy sample size


class Command(BaseCommand):
    help = "Train and evaluate the shortlist-prediction ML model on recruiter decisions."

    def handle(self, *args, **kwargs):
        labeled = Resume.objects.exclude(shortlisted=None)
        count = labeled.count()

        self.stdout.write(f"Found {count} labeled resumes (shortlisted or rejected).")

        if count < MIN_TO_TRAIN:
            self.stdout.write(self.style.ERROR(
                f"Need at least {MIN_TO_TRAIN} labeled resumes to train anything. "
                "Go mark some candidates as Shortlisted/Rejected on their detail "
                "pages first, then re-run this command."
            ))
            return

        if count < RECOMMENDED_MIN:
            self.stdout.write(self.style.WARNING(
                f"Only {count} labeled resumes — training anyway, but accuracy "
                f"will improve a lot once you have {RECOMMENDED_MIN}+ decisions "
                "recorded across different candidates."
            ))

        X, y = [], []
        for r in labeled:
            # 🔥 Uses the SAME feature-building function as prediction time
            # (in views.py) — critical so the model always sees data in
            # exactly the same shape it was trained on.
            X.append(build_feature_vector(
                job=r.job,
                skills_ai=r.skills_score or 0,
                exp_ai=r.experience_score or 0,
                edu_ai=r.education_score or 0,
                skill_match_ratio=r.skill_match_ratio or 0,
                matched_skills_count=r.matched_skills_count or 0,
                years_experience=r.years_experience or 0,
                education_level=r.education_level or 0,
                resume_word_count=r.resume_word_count or 0
            ))
            y.append(1 if r.shortlisted else 0)

        X = np.array(X)
        y = np.array(y)

        class_counts = np.bincount(y)
        if len(class_counts) < 2 or min(class_counts) == 0:
            self.stdout.write(self.style.ERROR(
                "All labeled resumes are the same decision (all shortlisted, "
                "or all rejected). The model needs at least one example of "
                "each outcome to learn a meaningful boundary."
            ))
            return

        # =========================
        # 🔥 MODEL COMPARISON VIA CROSS-VALIDATION
        # =========================
        # Can't have more CV folds than examples in the smallest class
        n_splits = min(5, min(class_counts))

        candidates = {
            "Logistic Regression": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000)),
            ]),
            "Random Forest": RandomForestClassifier(
                n_estimators=100, max_depth=4, random_state=42
            ),
        }

        if n_splits < 2:
            self.stdout.write(self.style.WARNING(
                "Too few examples in the smaller class to cross-validate "
                "reliably — training Logistic Regression directly without "
                "an evaluation score. Add more decisions for a real evaluation."
            ))
            best_name = "Logistic Regression"
            best_model = candidates[best_name]
            best_score = None   # unknown — no CV was possible
        else:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

            self.stdout.write("\n📊 Model comparison (cross-validated accuracy):")
            best_name, best_model, best_score = None, None, -1
            for name, model in candidates.items():
                scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
                mean_score = scores.mean()
                self.stdout.write(f"   {name}: {mean_score:.1%} (+/- {scores.std():.1%})")
                if mean_score > best_score:
                    best_score = mean_score
                    best_name = name
                    best_model = model

            self.stdout.write(self.style.SUCCESS(
                f"\n✅ Best model: {best_name} ({best_score:.1%} cross-validated accuracy)\n"
            ))

            # =========================
            # 🔥 DETAILED EVALUATION REPORT
            # =========================
            # cross_val_predict gives predictions made only when a resume
            # was in the held-out fold — never predicting on data the
            # model was trained on, so this is an honest report.
            y_pred_cv = cross_val_predict(best_model, X, y, cv=cv)

            self.stdout.write("📄 Classification Report (cross-validated, held-out predictions):")
            self.stdout.write(classification_report(
                y, y_pred_cv, target_names=["Rejected", "Shortlisted"], zero_division=0
            ))

            cm = confusion_matrix(y, y_pred_cv)
            self.stdout.write("Confusion Matrix:")
            self.stdout.write("                 Predicted Rejected   Predicted Shortlisted")
            self.stdout.write(f"Actual Rejected        {cm[0][0]:^15} {cm[0][1]:^18}")
            self.stdout.write(f"Actual Shortlisted     {cm[1][0]:^15} {cm[1][1]:^18}\n")

        # Refit the chosen model on ALL labeled data for the final saved model
        best_model.fit(X, y)

        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(best_model, MODEL_PATH)

        # 🔥 Save metadata so views.py knows how much to trust this model.
        # This is what fixes the "same resume, wildly different score"
        # problem — the model only gets full trust once it's proven
        # itself on enough data.
        meta = {
            "n_samples": int(count),
            "cv_accuracy": float(best_score) if best_score is not None else None,
            "model_name": best_name,
        }
        with open(MODEL_META_PATH, "w") as f:
            json.dump(meta, f)

        self.stdout.write(self.style.SUCCESS(
            f"Model trained on {count} labeled resumes and saved to:\n{MODEL_PATH}\n"
            "New resume uploads will now use this model according to how much "
            "data/accuracy it has earned (see trust tiers in views.py). "
            "Re-run this command anytime after recording more decisions to "
            "keep improving it."
        ))