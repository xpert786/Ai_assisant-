from django.db import models
from django.utils import timezone

class Company(models.Model):
    """Model to store company/account information"""
    COMPANY_TYPES = [
        ('llc', 'LLC'),
        ('corporation', 'Corporation'),
        ('partnership', 'Partnership'),
        ('sole_proprietorship', 'Sole Proprietorship'),
        ('nonprofit', 'Nonprofit'),
        ('other', 'Other'),
    ]
    
    STATUS_CHOICES = [
        ('active', 'Active'),
        ('inactive', 'Inactive'),
        ('pending', 'Pending'),
        ('suspended', 'Suspended'),
    ]
    
    # Core company information
    company_name = models.CharField(max_length=255, verbose_name="Company_Name")
    company_type = models.CharField(max_length=50, choices=COMPANY_TYPES, verbose_name="Type")
    state_of_formation = models.CharField(max_length=100, verbose_name="State_of_Formation")
    ein = models.CharField(max_length=20, blank=True, null=True, verbose_name="EIN")
    
    # Client associations
    associated_client = models.CharField(max_length=255, blank=True, null=True, verbose_name="Associated_Client")
    clients = models.TextField(blank=True, null=True, verbose_name="Clients")
    
    # Status and metadata
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='active', verbose_name="Status")
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    
    # Airtable integration
    airtable_id = models.CharField(max_length=255, blank=True, null=True, verbose_name="Airtable ID")
    
    class Meta:
        verbose_name = "Company"
        verbose_name_plural = "Companies"
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.company_name} - {self.get_company_type_display()}"
from django.db import models
from django.utils import timezone

class Message(models.Model):
    """Model to store messages and their AI responses"""
    PENDING = 'pending'
    PROCESSED = 'processed'
    APPROVED = 'approved'
    
    STATUS_CHOICES = [
        (PENDING, 'Pending'),
        (PROCESSED, 'Processed'),
        (APPROVED, 'Approved'),
    ]
    
    # Input content
    content = models.TextField(null=True, blank=True)
    image = models.ImageField(upload_to='message_images/', null=True, blank=True)
    extracted_content = models.TextField(null=True, blank=True, verbose_name="Extracted Text")
    
    # Client information
    client_name = models.CharField(max_length=255, null=True, blank=True)
    client_contact = models.CharField(max_length=255, null=True, blank=True)  # Phone/Email/Username
    
    # Response data
    ai_response = models.TextField(null=True, blank=True, verbose_name="AI Response")
    edited_response = models.TextField(null=True, blank=True, verbose_name="Edited Response")
    final_response = models.TextField(null=True, blank=True, verbose_name="Final Response")
    
    # Follow-up questions and conversation history
    follow_up_question = models.TextField(null=True, blank=True, verbose_name="Follow-up Question")
    conversation_history = models.JSONField(null=True, blank=True, verbose_name="Conversation History")
    parent_message = models.ForeignKey('self', null=True, blank=True, on_delete=models.SET_NULL, related_name='follow_ups')
    
    # Knowledge base and Airtable integration
    from_knowledge_base = models.BooleanField(default=False)
    knowledge_base_id = models.CharField(max_length=255, null=True, blank=True)
    saved_to_knowledge_base = models.BooleanField(default=False)
    saved_to_airtable = models.BooleanField(default=False)
    airtable_lead_id = models.CharField(max_length=255, null=True, blank=True)
    
    # ✅ New field for WhatsApp / Telegram / Gmail source
    source = models.CharField(max_length=50, null=True, blank=True, default="Manual", verbose_name="Message Source")
    
    # Metadata
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=PENDING)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        if self.client_name:
            return f"{self.client_name} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"
        elif self.client_contact:
            return f"{self.client_contact} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"
        elif self.image:
            return f"Image Upload - {self.created_at.strftime('%Y-%m-%d %H:%M')}"
        elif self.content and len(self.content) > 20:
            return f"{self.content[:20]}... - {self.created_at.strftime('%Y-%m-%d %H:%M')}"
        else:
            return f"Message - {self.created_at.strftime('%Y-%m-%d %H:%M')}"

class OpenAIConfig(models.Model):
    """Model to store OpenAI API configuration"""
    api_key = models.CharField(max_length=255, verbose_name="OpenAI API Key")
    model = models.CharField(max_length=100, default="gpt-3.5-turbo", verbose_name="OpenAI Model")
    temperature = models.FloatField(default=0.7, verbose_name="Temperature")
    max_tokens = models.IntegerField(default=1000, verbose_name="Max Tokens")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    
    def __str__(self):
        return f"OpenAI Config - {self.model} - {self.updated_at.strftime('%Y-%m-%d')}"
    
    class Meta:
        verbose_name = "OpenAI Configuration"
        verbose_name_plural = "OpenAI Configurations"

