from django.contrib import admin
from .models import Message

@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ('id', 'client_name', 'client_contact', 'status', 'created_at')
    list_filter = ('status', 'from_knowledge_base', 'saved_to_knowledge_base', 'saved_to_airtable')
    search_fields = ('content', 'extracted_content', 'ai_response', 'client_name', 'client_contact')
    readonly_fields = ('created_at', 'updated_at')
    fieldsets = (
        ('Client Information', {
            'fields': ('client_name', 'client_contact')
        }),
        ('Input Content', {
            'fields': ('content', 'image', 'extracted_content')
        }),
        ('Response Data', {
            'fields': ('ai_response', 'edited_response', 'final_response')
        }),
        ('Integration Status', {
            'fields': ('from_knowledge_base', 'knowledge_base_id', 'saved_to_knowledge_base', 
                      'saved_to_airtable', 'airtable_lead_id')
        }),
        ('Metadata', {
            'fields': ('status', 'created_at', 'updated_at')
        }),
    )
