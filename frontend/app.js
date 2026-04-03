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
        startProcessing(file.name);
    };

    // --- 2. Processing (Mock) Logic ---
    const startProcessing = (filename) => {
        switchView('processing');
        headerStatus.classList.add('processing');
        statusText.textContent = 'Processing Document...';

        const logs = [
            `> Found document: ${filename}`,
            `> Executing ingest.py...`,
            `> Extracting PDF content...`,
            `> Chunking text (Chunk size: 1000)...`,
            `> Generating vector embeddings via SentenceTransformers...`,
            `> Indexing into vector database...`,
            `> Ingestion complete!`,
            `> Starting ragchat.py engine...`,
            `> Loading LLM context...`,
            `> System ready.`
        ];

        let currentLogIndex = 0;
        terminalLogs.innerHTML = ''; // clear initial log
        
        const logInterval = setInterval(() => {
            if(currentLogIndex < logs.length) {
                const logEl = document.createElement('div');
                logEl.textContent = logs[currentLogIndex];
                terminalLogs.appendChild(logEl);
                terminalLogs.scrollTop = terminalLogs.scrollHeight;
                
                // Update Progress bar roughly based on index
                const progressPct = ((currentLogIndex + 1) / logs.length) * 100;
                progressFill.style.width = `${progressPct}%`;

                // At halfway, switch to ragchat title
                if (currentLogIndex === 7) {
                    processTitle.textContent = 'Starting ragchat.py...';
                    processSubtitle.textContent = 'Initalizing retrieval-augmented generation engine';
                }

                currentLogIndex++;
            } else {
                clearInterval(logInterval);
                setTimeout(() => {
                    finishProcessing();
                }, 1000); // 1s pause before switching to chat
            }
        }, 800); // 800ms between mock logs
    };

    const finishProcessing = () => {
        headerStatus.classList.remove('processing');
        headerStatus.classList.add('ready');
        statusText.textContent = 'System Ready';
        switchView('chat');
    };


    // --- 3. Chat Logic (Mock) ---
    const addMessage = (text, sender) => {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${sender}-message`;
        
        // Create avatar based on sender
        const avatarIcon = sender === 'bot' ? '<i class="fa-solid fa-robot"></i>' : '<i class="fa-solid fa-user"></i>';
        
        // Use marked/markdown parsing if real app, just safe text content here
        const bubbleContent = document.createElement('div');
        bubbleContent.className = 'bubble';
        bubbleContent.textContent = text; // helps prevent XSS too

        msgDiv.innerHTML = `<div class="avatar">${avatarIcon}</div>`;
        msgDiv.appendChild(bubbleContent);

        chatHistory.appendChild(msgDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;
    };

    const handleSend = () => {
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

        // Mock bot response
        setTimeout(() => {
            const typingEl = document.getElementById(typingId);
            if (typingEl) {
                typingEl.remove();
            }
            // Generate a fake but relevant-sounding response
            const response = generateMockResponse(text);
            addMessage(response, 'bot');
        }, 1500 + Math.random() * 1500); // Random delay between 1.5s - 3s
    };

    const generateMockResponse = (query) => {
        const lowerQ = query.toLowerCase();
        if (lowerQ.includes("who") || lowerQ.includes("what")) {
            return `Based on the document context, that specific detail highlights the core principles discussed. Specifically, the text notes the importance of structured reasoning across modules.`;
        } else if (lowerQ.includes("summarize") || lowerQ.includes("summary")) {
             return `The overall summary of the document points to three main things: 1. Effective utilization of vector embeddings. 2. Scalable RAG structures. 3. End-to-end reliability.`;
        } else {
             return `Indeed. According to the uploaded knowledge base, the system extracts related snippets using a similarity search, and processes your query to provide this accurate synthesis.`;
        }
    };

    sendBtn.addEventListener('click', handleSend);
    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            handleSend();
        }
    });
});
