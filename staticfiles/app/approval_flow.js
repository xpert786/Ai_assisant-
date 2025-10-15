/**
 * Approval Flow JavaScript
 * Handles the approval workflow for saving messages to Airtable
 */

class ApprovalFlow {
    constructor() {
        this.setupEventListeners();
    }

    setupEventListeners() {
        // Handle save buttons that require approval
        document.querySelectorAll('[data-requires-approval]').forEach(button => {
            button.addEventListener('click', (e) => {
                console.log('Approval flow button clicked:', e.target);
                e.preventDefault();
                this.handleSaveRequest(e.target);
            });
        });
    }

    async handleSaveRequest(button) {
        const messageId = button.dataset.messageId;
        const saveType = button.dataset.saveType || 'knowledge_base';
        
        console.log('handleSaveRequest called with:', { messageId, saveType });
        
        try {
            // Show loading state
            this.setButtonLoading(button, true);
            
            // Request approval
            const response = await fetch(`/messages/${messageId}/request-approval/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': this.getCSRFToken()
                },
                body: JSON.stringify({ type: saveType })
            });
            let data;
            try {
                data = await response.json();
            } catch (e) {
                const txt = await response.text();
                throw new Error(`Server returned ${response.status}. ${txt.slice(0,200)}`);
            }
            
            if (data.success && data.requires_approval) {
                // Show approval dialog
                // Attach desired save type
                data.preview.type = saveType;
                this.showApprovalDialog(data.preview, messageId, button);
            } else {
                // Handle error with details if any
                const detail = data.airtable_error ? `\nDetails: ${data.airtable_error}` : '';
                this.showError((data.message || 'Failed to request approval') + detail);
            }
        } catch (error) {
            console.error('Error requesting approval:', error);
            this.showError('Network error occurred');
        } finally {
            this.setButtonLoading(button, false);
        }
    }

    showApprovalDialog(preview, messageId, button) {
        const modalElement = document.getElementById('confirmSaveModal');
        const modal = new bootstrap.Modal(modalElement, { backdrop: true, keyboard: true });

        // Update modal content
        document.getElementById('confirmSaveModalLabel').textContent = "Confirm Save Operation";
        document.getElementById('modalActionDescription').textContent = `Action: ${preview.message}`;
        document.getElementById('modalClientName').textContent = preview.client_name || 'Not provided';
        document.getElementById('modalClientContact').textContent = preview.client_contact || 'Not provided';
        document.getElementById('modalExistingClient').textContent = preview.existing_client ? 'Yes' : 'No';
        document.getElementById('modalWillBeSavedTo').textContent = preview.existing_client ? 'Knowledge_Base (existing client)' : 'Leads (new client)';
        document.getElementById('modalMessageContent').innerHTML = this.escapeHtml(preview.client_message);
        document.getElementById('modalResponseContent').innerHTML = this.escapeHtml(preview.response_content);

        // Set up confirm button handler
        const confirmBtn = document.getElementById('confirmSaveModalBtn');
        confirmBtn.onclick = () => {
            this.confirmSave(preview, messageId, modalElement, button);
        };

        // Ensure cancel closes cleanly and restores UI
        const cancelBtn = modalElement.querySelector('[data-bs-dismiss="modal"].btn.btn-secondary');
        if (cancelBtn) {
            cancelBtn.onclick = () => {
                // Restore original button state in case it remained loading
                this.setButtonLoading(button, false);
                // Remove left-over backdrops if any
                setTimeout(() => {
                    document.querySelectorAll('.modal-backdrop').forEach(b => b.remove());
                    document.body.classList.remove('modal-open');
                }, 0);
            };
        }

        // Also on any hide event, cleanup backdrops and restore button
        modalElement.addEventListener('hidden.bs.modal', () => {
            this.setButtonLoading(button, false);
            document.querySelectorAll('.modal-backdrop').forEach(b => b.remove());
            document.body.classList.remove('modal-open');
        }, { once: true });

        modal.show();
    }

    async confirmSave(preview, messageId, modalElement, button) {
        try {
            const confirmButton = document.getElementById('confirmSaveModalBtn');
            this.setButtonLoading(confirmButton, true);
            
            // Determine server action from preview prepared by backend
            const action = preview.action || 'save_to_knowledge_base';

            const response = await fetch(`/messages/${messageId}/confirm-save/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': this.getCSRFToken()
                },
                body: JSON.stringify({
                    confirmed: true,
                    action: action
                })
            });
            let data;
            try {
                data = await response.json();
            } catch (e) {
                const txt = await response.text();
                throw new Error(`Server returned ${response.status}. ${txt.slice(0,200)}`);
            }
            
            if (data.success) {
                this.showSuccess(data.message);
                this.closeModal(modalElement);
                // Optionally refresh the page or update UI
                setTimeout(() => location.reload(), 1000);
            } else {
                // If KB blocked because still a lead, surface message and don't proceed
                const detail = data.airtable_error ? `\nDetails: ${data.airtable_error}` : '';
                const fields = data.fields_sent ? `\nFields: ${JSON.stringify(data.fields_sent)}` : '';
                this.showError((data.message || 'Failed to save') + detail + fields);
            }
        } catch (error) {
            console.error('Error confirming save:', error);
            this.showError(error?.message || 'Network error occurred');
        } finally {
            const confirmButton = document.getElementById('confirmSaveModalBtn');
            this.setButtonLoading(confirmButton, false);
        }
    }

    closeModal(modal) {
        const bsModal = bootstrap.Modal.getInstance(modal);
        if (bsModal) bsModal.hide();
    }

    setButtonLoading(button, loading) {
        if (!button) return; // guard
        if (loading) {
            button.disabled = true;
            button.dataset.originalText = button.textContent || 'Confirm & Save';
            button.textContent = 'Loading...';
        } else {
            button.disabled = false;
            if (button.dataset.originalText) {
                button.textContent = button.dataset.originalText;
                delete button.dataset.originalText;
            }
        }
    }

    showSuccess(message) {
        // Use Bootstrap success toast
        const successToast = document.getElementById('successToast');
        document.getElementById('successToastMessage').textContent = message;
        const toast = new bootstrap.Toast(successToast, { delay: 3000 });
        toast.show();
    }

    showError(message) {
        // Use Bootstrap error toast
        const errorToast = document.getElementById('errorToast');
        document.getElementById('errorToastMessage').textContent = message;
        const toast = new bootstrap.Toast(errorToast, { delay: 5000 });
        toast.show();
    }
    
    // Removed custom showNotification and addModalStyles as Bootstrap modals/toasts are used

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    getCSRFToken() {
        return document.querySelector('[name=csrfmiddlewaretoken]')?.value || 
               document.querySelector('meta[name="csrf-token"]')?.getAttribute('content') || '';
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new ApprovalFlow();
});
