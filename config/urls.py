from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('apps.rag_bot.urls')),         # home page
    path('api/rag/', include('apps.rag_bot.urls')),
    path('api/helpdesk/', include('apps.helpdesk_bot.urls')),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)