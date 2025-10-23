"""
URL configuration for n8n project.
"""
from django.contrib import admin
from django.urls import path, include
from django.conf.urls.static import static
from django.conf import settings
from django.conf.urls.static import static
from app import views
from app.views import chat_history_api
from app.views import chat_detail_api
from app.views import get_message_json
from app.views import ocr_ai_image_upload
from app.views import ocr_ai_image_upload


urlpatterns = [
    path('admin/', admin.site.urls),
    path('ai/', views.DashboardView.as_view(), name='dashboard'),
    path('ai/messages/create/', views.MessageCreateView.as_view(), name='message_create'),
    path('ai/messages/<int:pk>/edit/', views.MessageUpdateView.as_view(), name='message_edit'),
    path('ai/messages/<int:pk>/', views.MessageDetailView.as_view(), name='message_detail'),
    path('ai/messages/<int:pk>/save-to-kb/', views.save_to_knowledge_base, name='save_to_kb'),
    path('ai/messages/<int:pk>/create-lead/', views.create_lead, name='create_lead'),
    path('ai/messages/<int:message_id>/delete/', views.delete_message, name='delete_specific_message'),
    # path('ai/messages/<int:message_id>/process/', views.process_response, name='process_response'),
    path('ai/messages/<int:message_id>/pdf/', views.generate_pdf, name='generate_pdf'),
    path('ai/api/approve-and-save/', views.approve_and_save, name='approve_and_save'),
    path('ai/messages/<int:pk>/ajax-enhance/', views.ajax_enhance_response, name='ajax_enhance_response'),
    path('ai/messages/<int:pk>/json/', get_message_json, name='get_message_json'),
    path('ai/messages/<int:message_id>/save-to-airtable-accounts/', views.save_to_airtable_accounts, name='save_to_airtable_accounts'),
    path('ai/messages/<int:message_id>/update/', views.update_message, name='update_message'),
    path('ai/api/ocr-ai-image-upload/', ocr_ai_image_upload, name='ocr_ai_image_upload'),
    
    # Include app URLs for real-time chat functionality
    path('ai/', include('app.urls')),
]

# Add media URL configuration - this is needed for image uploads to work
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

urlpatterns += [
    path('ai/chat-history/', chat_history_api, name='chat_history_api'),
    path('ai/chat-detail/<int:pk>/', chat_detail_api, name='chat_detail_api'),
]
