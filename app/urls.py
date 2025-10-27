from django.urls import path
from . import views
from app.views import update_message, ocr_ai_image_upload, stream_ai_response, real_time_chat, real_time_chat_view


urlpatterns = [
    # Real-time AI endpoints
    path('api/stream-ai-response/', stream_ai_response, name='stream_ai_response'),
    path('api/real-time-chat/', real_time_chat, name='real_time_chat'),
    path('real-time-chat/', real_time_chat_view, name='real_time_chat_view'),
    
    # Client workflow endpoints
    path('messages/<int:pk>/handle-client-action/', views.handle_client_action, name='handle_client_action'),
    path('api/search-clients/', views.search_clients, name='search_clients'),
    path('api/companies/', views.get_companies, name='get_companies'),
    path('messages/<int:message_id>/save-to-airtable-accounts/', views.save_to_airtable_accounts, name='save_to_airtable_accounts'),
    path('messages/<int:pk>/save-to-kb/', views.save_to_knowledge_base, name='save_to_kb'),
    
    # Other endpoints
    path('messages/<int:message_id>/update/', update_message, name='update_message'),
    path('api/ocr-ai-image-upload/', ocr_ai_image_upload, name='ocr_ai_image_upload'),
] 