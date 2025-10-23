// Functions to handle duplicate knowledge base entries and duplicate leads

function showDuplicateKnowledgeBasePopup(message, entry) {
    const modalBody = document.getElementById('duplicateKnowledgeBaseModalBody');
    if (modalBody) {
        modalBody.innerHTML = `
            <div class="alert alert-warning">
                <i class="bi bi-exclamation-triangle me-2"></i>
                ${message}
            </div>
            ${entry ? `
                <div class="existing-entry-info">
                    <h6>Existing Entry Details:</h6>
                    <p><strong>Title:</strong> ${entry.title || 'N/A'}</p>
                    <p><strong>Category:</strong> ${entry.category || 'N/A'}</p>
                    <p><strong>Last Updated:</strong> ${entry.last_updated || 'N/A'}</p>
                </div>
            ` : ''}
        `;
        
        const modal = new bootstrap.Modal(document.getElementById('duplicateKnowledgeBaseModal'));
        modal.show();
    } else {
        alert(message);
    }
}

function showDuplicateLeadPopup(message, client) {
    const modalBody = document.getElementById('duplicateLeadModalBody');
    if (modalBody) {
        modalBody.innerHTML = `
            <div class="alert alert-warning">
                <i class="bi bi-exclamation-triangle me-2"></i>
                ${message}
            </div>
            ${client ? `
                <div class="existing-client-info">
                    <h6>Existing Lead Details:</h6>
                    <p><strong>Name:</strong> ${client.name || 'N/A'}</p>
                    <p><strong>Contact:</strong> ${client.contact || 'N/A'}</p>
                    <p><strong>Status:</strong> ${client.status || 'N/A'}</p>
                </div>
            ` : ''}
        `;
        
        const modal = new bootstrap.Modal(document.getElementById('duplicateLeadModal'));
        modal.show();
    } else {
        alert(message);
    }
}

// Enhanced save handlers with duplicate detection
function handleSaveToKnowledgeBase(messageId, button) {
    const originalText = button.innerHTML;
    
    fetch(`/ai/messages/${messageId}/save-to-kb/`, {
        method: 'POST',
        headers: {
            'X-CSRFToken': getCSRFToken(),
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({})
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showSuccessToast('Successfully saved to Airtable');
            button.classList.remove('btn-outline-accent');
            button.classList.add('btn-success');
            button.innerHTML = '<i class="bi bi-check me-1"></i>Saved';
            button.disabled = true;
        } else {
            // Handle duplicate knowledge base entry
            if (data.is_duplicate && data.message === 'This knowledge base entry already exists') {
                showDuplicateKnowledgeBasePopup(data.message, data.existing_entry || {});
            } else {
                showErrorToast('Error: ' + data.message);
            }
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showErrorToast('An error occurred while saving to Airtable');
    });
}