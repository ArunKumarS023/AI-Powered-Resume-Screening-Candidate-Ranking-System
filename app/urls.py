from django.urls import path
from . import views
from django.contrib.auth import views as auth_views
urlpatterns = [

    path('', views.home, name='home'),
    path('create-job/', views.create_job, name='create_job'),
    path('upload/<int:job_id>/', views.upload_resume, name='upload_resume'),
    path('analytics/', views.analytics, name='analytics'),
    path('resume/<int:id>/', views.resume_detail, name='resume_detail'),
    path('jobs/', views.jobs, name='jobs'),
    path('download/<int:job_id>/', views.download_csv),
    # 🔥 NEW (Job Details Page)
    path('job/<int:id>/', views.job_detail, name='job_detail'),

    # 🔥 NEW: records real shortlist/reject decisions used to train the ML model
    path('resume/<int:id>/shortlist/', views.mark_shortlist, name='mark_shortlist'),

]