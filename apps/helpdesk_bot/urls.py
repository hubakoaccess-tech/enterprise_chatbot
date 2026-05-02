from django.urls import path
from . import views

urlpatterns = [
    path('chat/', views.helpdesk_chat, name='helpdesk_chat'),
]