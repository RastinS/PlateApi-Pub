from django.urls import path, include

from .views import *

urlpatterns = [
    path('detect/', FileUploadView.as_view()),
]