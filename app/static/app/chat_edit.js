function setupChatEdit(li, chat) {
    const editButton = li.querySelector('.chat-edit-item');
    const chatSpan = li.querySelector('.chat-content');
    const dropdownMenu = li.querySelector('.dropdown-menu');

    editButton.addEventListener('click', function(e) {
        e.preventDefault();
        e.stopPropagation();

        const currentText = chatSpan.textContent;
        
        // Create input field
        const input = document.createElement('input');
        input.type = 'text';
        input.value = currentText;
        input.className = 'form-control chat-edit-input';
        input.style.cssText = `
            background-color: #2a2a2a;
            color: #fff;
            border: 1px solid #4cc9f0;
            border-radius: 4px;
            padding: 5px 10px;
            width: calc(100% - 40px);
        `;

        // Replace content with input
        chatSpan.style.display = 'none';
        chatSpan.parentNode.insertBefore(input, chatSpan);
        input.focus();
        
        // Close dropdown
        dropdownMenu.style.display = 'none';

        // Function to save changes
        function saveChanges() {
            const newText = input.value.trim();
            if (!newText) {
                input.remove();
                chatSpan.style.display = '';
                return;
            }

            fetch(`/ai/messages/${chat.id}/update/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    content: newText
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    chatSpan.textContent = newText;
                    input.remove();
                    chatSpan.style.display = '';
                } else {
                    alert('Failed to save changes: ' + (data.message || 'Unknown error'));
                    input.remove();
                    chatSpan.style.display = '';
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Failed to save changes');
                input.remove();
                chatSpan.style.display = '';
            });
        }

        // Save on Enter
        input.addEventListener('keydown', function(event) {
            if (event.key === 'Enter') {
                event.preventDefault();
                saveChanges();
            } else if (event.key === 'Escape') {
                input.remove();
                chatSpan.style.display = '';
            }
        });

        // Save on blur
        input.addEventListener('blur', function() {
            saveChanges();
        });
    });
}
