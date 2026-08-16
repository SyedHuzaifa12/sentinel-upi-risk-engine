from django.urls import path

from .views import auth, monitoring, pages, prediction, reports, review

urlpatterns = [
    path('review/', review.review_queue, name='review_queue'),
    path('monitoring/', monitoring.monitoring_dashboard, name='monitoring_dashboard'),
    path('', auth.home, name='users-home'),
    path('register/', auth.RegisterView.as_view(), name='users-register'),
    path('profile/', auth.profile, name='users-profile'),
    path('logout_view/', auth.logout_view, name='logout_view'),
    path('index/', auth.index, name='users-index'),
    path('Basic_report/', reports.Basic_report, name='Basic_report'),
    path('Metrics_report/', reports.Metrics_report, name='Metrics_report'),
    path('profile_list/', pages.profile_list, name='profile_list'),

    path('awareness/', pages.awareness_page, name='awareness'),

    path('prediction/', prediction.prediction_view, name='prediction'),
    path('analyze_upi/', prediction.analyze_upi, name='analyze_upi'),
    path('database/', prediction.model_db_view, name='database'),
]


 