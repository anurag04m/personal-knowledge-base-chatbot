document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements ---
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    
    const views = {
        upload: document.getElementById('upload-view'),
        processing: document.getElementById('processing-view'),
        chat: document.getElementById('chat-view')
    };

    const headerStatus = document.getElementById('header-status');
    const statusText = headerStatus.querySelector('.text');
    const clearChatBtn = document.getElementById('clear-chat-btn');
    
    // Processing Elements
    const terminalLogs = document.getElementById('terminal-logs');
    const progressFill = document.getElementById('progress-fill');
    const processTitle = document.getElementById('process-title');
    const processSubtitle = document.getElementById('process-subtitle');

    // Chat Elements
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    const chatHistory = document.getElementById('chat-history');

    // --- State Management ---
    const switchView = (viewName) => {
        // Hide all views
        Object.values(views).forEach(v => {
            v.classList.remove('active');
            // Wait for opacity transition before fully hiding
            setTimeout(() => {
                if(!v.classList.contains('active')) {
                    v.classList.add('hidden');
                }
            }, 500);
        });

        // Show target view
        const target = views[viewName];
        target.classList.remove('hidden');
        // Small delay to allow display flex to apply before opacity transition
        setTimeout(() => {
            target.classList.add('active');
        }, 50);
    };

    // --- 1. Upload Logic ---
    browseBtn.addEventListener('click', () => {
        fileInput.click();
    });

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length) {
            handleFileSelect(fileInput.files[0]);
        }
    });

    const handleFileSelect = (file) => {
        if(file.type !== 'application/pdf' && !file.name.endsWith('.pdf')) {
            alert('Please select a PDF document.');
            return;
        }
        
        // Start Processing Flow
        startProcessing(file);
    };

    // --- 2. Processing (Actual) Logic ---
    const startProcessing = async (file) => {
        switchView('processing');
        headerStatus.classList.add('processing');
        statusText.textContent = 'Processing Document...';

        terminalLogs.innerHTML = ''; // clear initial log
        
        const logEl = document.createElement('div');
        logEl.textContent = `> Uploading and ingesting ${file.name}... (This may take a minute)`;
        terminalLogs.appendChild(logEl);

        // Update progress bar to show activity
        progressFill.style.width = '50%';
        processTitle.textContent = 'Processing...';
        processSubtitle.textContent = 'Generating vector embeddings and ingesting data';

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('http://localhost:5000/upload', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            
            if(response.ok) {
                const successEl = document.createElement('div');
                successEl.textContent = `> ✅ ${data.message}`;
                terminalLogs.appendChild(successEl);
                terminalLogs.scrollTop = terminalLogs.scrollHeight;
                
                progressFill.style.width = '100%';
                
                setTimeout(() => {
                    finishProcessing();
                }, 1000); // 1s pause before switching to chat
            } else {
                alert('Upload failed: ' + data.error);
                switchView('upload');
                headerStatus.classList.remove('processing');
                statusText.textContent = 'Upload Document';
            }
        } catch (error) {
            alert('Connection to server failed. Ensure backend runs on port 5000.');
            switchView('upload');
            headerStatus.classList.remove('processing');
            statusText.textContent = 'Upload Document';
        }
    };

    const finishProcessing = () => {
        headerStatus.classList.remove('processing');
        headerStatus.classList.add('ready');
        statusText.textContent = 'System Ready';
        clearChatBtn.classList.remove('hidden'); // Show clear chat button
        switchView('chat');
    };


    // --- 3. Chat Logic (Mock) ---
    const addMessage = (text, sender) => {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${sender}-message`;
        
        // Create avatar based on sender
        const avatarIcon = sender === 'bot' ? '<i class="fa-solid fa-robot"></i>' : '<i class="fa-solid fa-user"></i>';
        
        const bubbleContent = document.createElement('div');
        bubbleContent.className = 'bubble markdown-body';
        
        if (sender === 'bot') {
            bubbleContent.innerHTML = marked.parse(text);
        } else {
            bubbleContent.textContent = text;
        }

        msgDiv.innerHTML = `<div class="avatar">${avatarIcon}</div>`;
        msgDiv.appendChild(bubbleContent);

        chatHistory.appendChild(msgDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;
    };

    const handleSend = async () => {
        const text = chatInput.value.trim();
        if (!text) return;

        // User message
        addMessage(text, 'user');
        chatInput.value = '';

        // Add a typing placeholder
        const typingId = 'typing-' + Date.now();
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message bot-message';
        typingDiv.id = typingId;
        typingDiv.innerHTML = `
            <div class="avatar"><i class="fa-solid fa-robot"></i></div>
            <div class="bubble">
                <div class="typing-indicator">
                    <span class="dot"></span>
                    <span class="dot"></span>
                    <span class="dot"></span>
                </div>
            </div>
        `;
        chatHistory.appendChild(typingDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;

        try {
            const response = await fetch('http://localhost:5000/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ query: text })
            });
            const data = await response.json();
            
            const typingEl = document.getElementById(typingId);
            if (typingEl) {
                typingEl.remove();
            }
            
            if (response.ok) {
                addMessage(data.answer, 'bot');
            } else {
                addMessage("Error: " + data.error, 'bot');
            }
        } catch (error) {
            const typingEl = document.getElementById(typingId);
            if (typingEl) {
                typingEl.remove();
            }
            addMessage("Error connecting to the backend server.", 'bot');
        }
    };

    sendBtn.addEventListener('click', handleSend);
    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            handleSend();
        }
    });

    clearChatBtn.addEventListener('click', () => {
        if (confirm('Are you sure you want to clear the chat history?')) {
            // Keep only the initial bot message
            const firstMessage = chatHistory.firstElementChild;
            chatHistory.innerHTML = '';
            if (firstMessage) {
                chatHistory.appendChild(firstMessage);
            }
        }
    });
});
