from django import forms
from .models import Message, OpenAIConfig, Company

class CompanyForm(forms.ModelForm):
    """Form for creating and editing company information"""
    class Meta:
        model = Company
        fields = ['company_name', 'company_type', 'state_of_formation', 'ein', 'associated_client', 'clients', 'status']
        widgets = {
            'company_name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter company name',
                'required': True
            }),
            'company_type': forms.Select(attrs={
                'class': 'form-select',
                'required': True
            }),
            'state_of_formation': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'e.g., Delaware, California, New York',
                'required': True
            }),
            'ein': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'XX-XXXXXXX (optional)',
                'maxlength': '20'
            }),
            'associated_client': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Associated client name or contact'
            }),
            'clients': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': 'List of clients or additional information'
            }),
            'status': forms.Select(attrs={
                'class': 'form-select'
            })
        }

class CompanySearchForm(forms.Form):
    """Form for searching companies"""
    search_query = forms.CharField(
        max_length=255,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Search by company name, EIN, or client...'
        })
    )
    
    company_type = forms.ChoiceField(
        choices=[('', 'All Types')] + Company.COMPANY_TYPES,
        required=False,
        widget=forms.Select(attrs={
            'class': 'form-select'
        })
    )
    
    status = forms.ChoiceField(
        choices=[('', 'All Statuses')] + Company.STATUS_CHOICES,
        required=False,
        widget=forms.Select(attrs={
            'class': 'form-select'
        })
    )

class MessageForm(forms.ModelForm):
    """Form for submitting new messages"""
    class Meta:
        model = Message
        fields = ['content', 'image']
        widgets = {
            'content': forms.Textarea(attrs={
                'rows': 5, 
                'class': 'form-control', 
                'placeholder': 'Type or paste your message here...',
                'required': False,
                'autofocus': True
            }),
            'image': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': 'image/*',
                'required': False,
                'style': 'height: 100%; width: 100%;'
            })
        }
        
    def clean(self):
        cleaned_data = super().clean()
        content = cleaned_data.get('content')
        image = cleaned_data.get('image')
        
        # Either content or image is required
        if not content and not image:
            raise forms.ValidationError("Please provide either a message or upload a screenshot.")
        
        return cleaned_data

class ResponseEditForm(forms.ModelForm):
    """Form for editing AI responses"""
    class Meta:
        model = Message
        fields = ['edited_response']
        widgets = {
            'edited_response': forms.Textarea(attrs={
                'rows': 8,
                'class': 'form-control',
                'placeholder': 'Edit the AI response here...'
            })
        }

class OpenAIConfigForm(forms.ModelForm):
    """Form for configuring OpenAI API settings"""
    class Meta:
        model = OpenAIConfig
        fields = ['api_key', 'model', 'temperature', 'max_tokens']
        widgets = {
            'api_key': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter your OpenAI API key',
                'autocomplete': 'off'
            }),
            'model': forms.Select(attrs={
                'class': 'form-select'
            }, choices=[
                ('gpt-3.5-turbo', 'GPT-3.5 Turbo'),
                ('gpt-4', 'GPT-4'),
                ('gpt-4-turbo', 'GPT-4 Turbo')
            ]),
            'temperature': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': '0',
                'max': '2',
                'step': '0.1'
            }),
            'max_tokens': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': '100',
                'max': '4000',
                'step': '100'
            })
        } 