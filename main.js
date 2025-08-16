/**
 * 🤖 AI Agent - Self-Made Intelligent Chatbot
 * Tích hợp tất cả components để tạo thành một AI agent hoàn chỉnh
 */

// Import all components (trong môi trường Node.js)
const { NeuralNetwork } = require('./core/brain.js');
const { TextProcessor, LanguageModel } = require('./core/nlp.js');
const { Memory, ReasoningEngine } = require('./core/memory.js');

class AIAgent {
    constructor(config = {}) {
        this.name = config.name || 'AI Assistant';
        this.personality = config.personality || 'helpful and friendly';
        this.language = config.language || 'vi';
        
        console.log(`🤖 Khởi tạo AI Agent: ${this.name}`);
        
        // Initialize components
        this.initializeComponents(config);
        
        // Training data
        this.trainingData = [];
        this.conversationHistory = [];
        
        // State
        this.isLearning = config.learning !== false;
        this.isTraining = false;
        this.responseLatency = [];
        
        console.log('✅ AI Agent đã sẵn sàng!');
    }

    // Khởi tạo các thành phần
    initializeComponents(config) {
        // Memory system
        this.memory = new Memory({
            shortTermCapacity: config.shortTermCapacity || 15,
            longTermCapacity: config.longTermCapacity || 2000,
            episodicCapacity: config.episodicCapacity || 1000
        });

        // NLP processor
        this.nlp = new TextProcessor();
        
        // Language model
        this.languageModel = new LanguageModel(this.nlp);
        
        // Reasoning engine
        this.reasoning = new ReasoningEngine(this.memory);
        
        // Neural network for response generation
        this.brain = new NeuralNetwork({
            layers: [100, 64, 32, 20], // Input, hidden layers, output
            activations: ['relu', 'relu', 'relu', 'sigmoid'],
            learningRate: 0.001
        });

        // Load initial knowledge
        this.loadInitialKnowledge();
    }

    // Load kiến thức ban đầu
    loadInitialKnowledge() {
        const basicKnowledge = [
            "Tôi là một AI agent thông minh được tạo ra để giúp đỡ con người.",
            "Tôi có thể học hỏi từ cuộc trò chuyện và cải thiện theo thời gian.",
            "Việt Nam là một đất nước xinh đẹp ở Đông Nam Á.",
            "AI là trí tuệ nhân tạo, giúp máy tính mô phỏng trí thông minh con người.",
            "JavaScript là ngôn ngữ lập trình phổ biến cho web development.",
            "Machine learning là một phần của AI giúp máy tính học từ dữ liệu.",
            "Python là ngôn ngữ lập trình được sử dụng nhiều trong AI và data science."
        ];

        basicKnowledge.forEach(knowledge => {
            this.memory.addToLongTerm(`knowledge_${Date.now()}_${Math.random()}`, knowledge, 0.8);
        });

        // Build vocabulary from basic knowledge
        this.nlp.buildVocabulary(basicKnowledge);
        
        // Build language model
        this.languageModel.buildModel(basicKnowledge);

        console.log('📚 Đã load kiến thức cơ bản');
    }

    // Xử lý tin nhắn đầu vào
    async processMessage(message, userContext = {}) {
        const startTime = Date.now();
        
        console.log(`👤 User: ${message}`);
        
        try {
            // 1. Preprocess input
            const processedInput = this.nlp.preprocess(message, {
                lowercase: true,
                removeStopwords: false,
                detectLanguage: true,
                extractEntities: true,
                analyzeSentiment: true,
                classifyIntent: true,
                extractKeywords: true
            });

            console.log(`🔍 Processed: ${JSON.stringify(processedInput.intent)}`);

            // 2. Add to memory
            this.memory.addToShortTerm({
                type: 'user_message',
                content: message,
                processed: processedInput,
                context: userContext,
                timestamp: Date.now()
            });

            // 3. Reasoning
            const reasoning = this.reasoning.reason(message);
            console.log(`🤔 Reasoning confidence: ${reasoning.confidence.toFixed(2)}`);

            // 4. Generate response
            const response = await this.generateResponse(processedInput, reasoning);

            // 5. Add response to memory
            this.memory.addToShortTerm({
                type: 'ai_response', 
                content: response.content,
                confidence: response.confidence,
                timestamp: Date.now()
            });

            // 6. Learn from interaction
            if (this.isLearning) {
                this.learnFromInteraction(message, response.content, processedInput);
            }

            // 7. Update conversation history
            this.conversationHistory.push({
                user: message,
                ai: response.content,
                timestamp: Date.now(),
                processing: processedInput,
                reasoning: reasoning
            });

            const endTime = Date.now();
            const latency = endTime - startTime;
            this.responseLatency.push(latency);

            console.log(`🤖 AI: ${response.content} (${latency}ms)`);

            return {
                content: response.content,
                confidence: response.confidence,
                sentiment: processedInput.sentiment,
                intent: processedInput.intent,
                entities: processedInput.entities,
                keywords: processedInput.keywords,
                reasoning: reasoning.conclusions,
                processingTime: latency
            };

        } catch (error) {
            console.error('❌ Error processing message:', error);
            return {
                content: "Xin lỗi, tôi gặp vấn đề khi xử lý tin nhắn của bạn. Bạn có thể thử lại không?",
                confidence: 0.5,
                error: true,
                processingTime: Date.now() - startTime
            };
        }
    }

    // Generate response
    async generateResponse(processedInput, reasoning) {
        let response = '';
        let confidence = 0.5;

        // 1. Intent-based responses
        if (processedInput.intent?.intent) {
            const intentResponse = this.generateIntentResponse(processedInput.intent.intent, processedInput);
            if (intentResponse) {
                response = intentResponse.content;
                confidence = intentResponse.confidence;
            }
        }

        // 2. Use reasoning conclusions
        if (!response && reasoning.conclusions.length > 0) {
            response = reasoning.conclusions[0];
            confidence = reasoning.confidence;
        }

        // 3. Memory-based response
        if (!response) {
            const memoryResults = this.memory.recall(processedInput.originalText);
            if (memoryResults.length > 0) {
                response = this.generateContextualResponse(memoryResults[0].content, processedInput);
                confidence = memoryResults[0].relevance;
            }
        }

        // 4. Pattern-based response
        if (!response) {
            response = this.generatePatternResponse(processedInput.originalText);
            confidence = 0.6;
        }

        // 5. Language model generation
        if (!response) {
            response = this.languageModel.generateText(processedInput.originalText, 20);
            confidence = 0.4;
        }

        // 6. Fallback response
        if (!response || response.trim().length === 0) {
            response = this.generateFallbackResponse(processedInput);
            confidence = 0.3;
        }

        // Personality adjustment
        response = this.applyPersonality(response);

        return {
            content: response.trim(),
            confidence: Math.max(0.1, Math.min(1.0, confidence))
        };
    }

    // Generate intent-based response
    generateIntentResponse(intent, processedInput) {
        const intentResponses = {
            greeting: {
                responses: [
                    "Xin chào! Tôi là AI assistant. Tôi có thể giúp gì cho bạn?",
                    "Chào bạn! Rất vui được gặp bạn. Hôm nay bạn cần hỗ trợ gì?",
                    "Hi! Tôi sẵn sàng trò chuyện với bạn về bất cứ điều gì."
                ],
                confidence: 0.9
            },
            question: {
                responses: [
                    "Đó là một câu hỏi hay! Để tôi suy nghĩ...",
                    "Tôi hiểu bạn muốn biết về điều này. Dựa vào kiến thức của tôi...",
                    "Theo như tôi biết..."
                ],
                confidence: 0.7
            },
            request: {
                responses: [
                    "Tôi sẽ cố gắng giúp bạn.",
                    "Để tôi xem tôi có thể hỗ trợ gì...",
                    "Tôi sẽ làm hết sức để giúp bạn."
                ],
                confidence: 0.8
            },
            goodbye: {
                responses: [
                    "Tạm biệt! Rất vui được trò chuyện với bạn.",
                    "Bye bye! Hẹn gặp lại bạn.",
                    "Chào tạm biệt! Chúc bạn một ngày tốt lành."
                ],
                confidence: 0.9
            },
            compliment: {
                responses: [
                    "Cảm ơn bạn! Tôi rất vui khi được khen.",
                    "Thank you! Tôi sẽ cố gắng làm tốt hơn.",
                    "Cảm ơn lời khen! Điều đó động viên tôi rất nhiều."
                ],
                confidence: 0.85
            }
        };

        const intentData = intentResponses[intent];
        if (intentData) {
            const randomResponse = intentData.responses[Math.floor(Math.random() * intentData.responses.length)];
            return {
                content: randomResponse,
                confidence: intentData.confidence
            };
        }

        return null;
    }

    // Generate contextual response
    generateContextualResponse(context, processedInput) {
        // Analyze context and generate appropriate response
        const sentiment = processedInput.sentiment?.sentiment;
        
        if (sentiment === 'positive') {
            return `Tôi hiểu! ${context} Đó là điều tích cực.`;
        } else if (sentiment === 'negative') {
            return `Tôi thấy bạn có vẻ không vui về điều này. ${context} Tôi có thể giúp gì không?`;
        } else {
            return `Về vấn đề này, ${context}`;
        }
    }

    // Generate pattern-based response
    generatePatternResponse(input) {
        const patterns = [
            {
                pattern: /tên (của )?bạn là gì|bạn tên gì/i,
                response: `Tôi là ${this.name}, một AI assistant thông minh. Rất vui được làm quen với bạn!`
            },
            {
                pattern: /bạn có thể làm gì|khả năng của bạn/i,
                response: "Tôi có thể trò chuyện, trả lời câu hỏi, giải toán, phân tích thông tin và học hỏi từ cuộc trò chuyện với bạn."
            },
            {
                pattern: /(\d+)\s*[\+\-\*\/]\s*(\d+)/,
                response: (match) => {
                    try {
                        const result = eval(match[0]);
                        return `Kết quả là: ${result}`;
                    } catch (e) {
                        return "Tôi không thể tính toán được phép tính này.";
                    }
                }
            },
            {
                pattern: /mấy giờ|thời gian|time/i,
                response: `Hiện tại là: ${new Date().toLocaleString('vi-VN')}`
            },
            {
                pattern: /cảm ơn|thanks|thank you/i,
                response: "Không có gì! Tôi rất vui được giúp đỡ bạn."
            }
        ];

        for (const { pattern, response } of patterns) {
            const match = input.match(pattern);
            if (match) {
                if (typeof response === 'function') {
                    return response(match);
                }
                return response;
            }
        }

        return null;
    }

    // Apply personality to response
    applyPersonality(response) {
        const personalityTraits = {
            'helpful and friendly': {
                prefixes: ['', 'Tôi nghĩ rằng ', 'Theo ý kiến của tôi, '],
                suffixes: ['', ' Bạn có cần thêm thông tin gì không?', ' Hy vọng điều này hữu ích!'],
                tone: 'warm'
            },
            'professional': {
                prefixes: ['Dựa trên phân tích, ', 'Kết quả cho thấy ', ''],
                suffixes: ['', ' Đây là câu trả lời chính xác nhất tôi có thể đưa ra.'],
                tone: 'formal'
            },
            'casual': {
                prefixes: ['', 'Mình nghĩ ', 'Theo mình thì '],
                suffixes: ['', ' Bạn thấy sao?', ' Có đúng không?'],
                tone: 'informal'
            }
        };

        const trait = personalityTraits[this.personality] || personalityTraits['helpful and friendly'];
        
        const prefix = trait.prefixes[Math.floor(Math.random() * trait.prefixes.length)];
        const suffix = trait.suffixes[Math.floor(Math.random() * trait.suffixes.length)];
        
        return prefix + response + suffix;
    }

    // Learn from interaction
    learnFromInteraction(userMessage, aiResponse, processedInput) {
        // Store successful patterns
        if (processedInput.sentiment?.sentiment === 'positive') {
            this.memory.addToLongTerm(
                `successful_pattern_${Date.now()}`,
                `User said: "${userMessage}" -> AI responded: "${aiResponse}" -> Positive feedback`,
                0.7
            );
        }

        // Learn new vocabulary
        if (processedInput.keywords && processedInput.keywords.length > 0) {
            processedInput.keywords.forEach(keyword => {
                this.nlp.vocabulary.set(keyword.word, this.nlp.vocabulary.size);
                const frequency = this.nlp.wordFrequency.get(keyword.word) || 0;
                this.nlp.wordFrequency.set(keyword.word, frequency + 1);
            });
        }

        // Add to episodic memory
        this.memory.addEpisodicMemory({
            type: 'conversation',
            userMessage,
            aiResponse,
            context: processedInput,
            outcome: 'completed'
        });

        // Update reasoning rules if applicable
        if (processedInput.intent?.confidence > 0.8) {
            this.reasoning.addRule(`learned_${Date.now()}`, {
                pattern: [userMessage],
                conclusion: aiResponse,
                confidence: processedInput.intent.confidence * 0.6
            });
        }
    }

    // Train the neural network
    async trainNetwork(trainingData = null) {
        if (this.isTraining) {
            console.log('⚠️ Training already in progress');
            return;
        }

        this.isTraining = true;
        console.log('🎓 Bắt đầu training neural network...');

        try {
            const data = trainingData || this.prepareTrainingData();
            
            if (data.inputs.length === 0) {
                console.log('⚠️ Không có dữ liệu training');
                return;
            }

            await this.brain.train(data.inputs, data.outputs, 500);
            
            // Evaluate performance
            const evaluation = this.brain.evaluate(data.inputs.slice(-10), data.outputs.slice(-10));
            console.log(`📊 Training completed - Accuracy: ${evaluation.accuracy.toFixed(1)}%`);

        } catch (error) {
            console.error('❌ Training error:', error);
        } finally {
            this.isTraining = false;
        }
    }

    // Prepare training data from conversation history
    prepareTrainingData() {
        const inputs = [];
        const outputs = [];

        this.conversationHistory.forEach(conversation => {
            // Convert text to vectors
            const inputVector = this.nlp.textToVector(conversation.user, 100);
            const outputVector = this.nlp.textToVector(conversation.ai, 20);

            if (inputVector.length > 0 && outputVector.length > 0) {
                inputs.push(inputVector);
                outputs.push(outputVector);
            }
        });

        return { inputs, outputs };
    }

    // Save AI Agent state
    save() {
        const agentState = {
            config: {
                name: this.name,
                personality: this.personality,
                language: this.language
            },
            memory: this.memory.save(),
            nlp: this.nlp.save(),
            reasoning: this.reasoning.save(),
            brain: this.brain.save(),
            conversationHistory: this.conversationHistory.slice(-100), // Last 100 conversations
            trainingData: this.trainingData,
            stats: this.getStats(),
            timestamp: Date.now(),
            version: '1.0.0'
        };

        return JSON.stringify(agentState, null, 2);
    }

    // Load AI Agent state
    load(agentStateData) {
        try {
            const state = typeof agentStateData === 'string' ? 
                JSON.parse(agentStateData) : agentStateData;

            // Load config
            if (state.config) {
                this.name = state.config.name || this.name;
                this.personality = state.config.personality || this.personality;
                this.language = state.config.language || this.language;
            }

            // Load components
            if (state.memory) this.memory.load(state.memory);
            if (state.nlp) this.nlp.load(state.nlp);
            if (state.reasoning) this.reasoning.load(state.reasoning);
            if (state.brain) this.brain.load(state.brain);

            // Load conversation history
            this.conversationHistory = state.conversationHistory || [];
            this.trainingData = state.trainingData || [];

            console.log('✅ AI Agent loaded successfully!');
            console.log(`📊 Loaded ${this.conversationHistory.length} conversations`);
            
        } catch (error) {
            console.error('❌ Error loading AI Agent state:', error);
        }
    }

    // Get statistics
    getStats() {
        return {
            agent: {
                name: this.name,
                personality: this.personality,
                uptime: Date.now(),
                totalConversations: this.conversationHistory.length,
                isLearning: this.isLearning,
                isTraining: this.isTraining
            },
            performance: {
                averageResponseTime: this.responseLatency.length > 0 ? 
                    (this.responseLatency.reduce((a, b) => a + b, 0) / this.responseLatency.length).toFixed(2) + 'ms' : '0ms',
                totalResponses: this.responseLatency.length,
                fastestResponse: Math.min(...this.responseLatency) + 'ms',
                slowestResponse: Math.max(...this.responseLatency) + 'ms'
            },
            memory: this.memory.getStats(),
            reasoning: this.reasoning.getStats(),
            nlp: {
                vocabularySize: this.nlp.vocabulary.size,
                totalWords: this.nlp.wordFrequency.size,
                bigramsCount: this.nlp.bigramFrequency.size
            },
            neural: {
                layers: this.brain.config.layers,
                trainingHistory: this.brain.trainingHistory.length,
                lastTraining: this.brain.trainingHistory.length > 0 ? 
                    new Date(this.brain.trainingHistory[this.brain.trainingHistory.length - 1].timestamp).toLocaleString() : 'Never'
            }
        };
    }

    // Reset AI Agent
    reset() {
        console.log('🔄 Resetting AI Agent...');
        
        this.conversationHistory = [];
        this.trainingData = [];
        this.responseLatency = [];
        
        this.memory = new Memory();
        this.reasoning = new ReasoningEngine(this.memory);
        
        this.loadInitialKnowledge();
        
        console.log('✅ AI Agent reset completed');
    }

    // Set personality
    setPersonality(personality) {
        const validPersonalities = ['helpful and friendly', 'professional', 'casual'];
        
        if (validPersonalities.includes(personality)) {
            this.personality = personality;
            console.log(`🎭 Personality changed to: ${personality}`);
        } else {
            console.log(`❌ Invalid personality. Valid options: ${validPersonalities.join(', ')}`);
        }
    }

    // Enable/disable learning
    toggleLearning() {
        this.isLearning = !this.isLearning;
        console.log(`🧠 Learning ${this.isLearning ? 'enabled' : 'disabled'}`);
        return this.isLearning;
    }

    // Get conversation history
    getConversationHistory(limit = 10) {
        return this.conversationHistory.slice(-limit);
    }

    // Search conversations
    searchConversations(query) {
        return this.conversationHistory.filter(conv => 
            conv.user.toLowerCase().includes(query.toLowerCase()) ||
            conv.ai.toLowerCase().includes(query.toLowerCase())
        );
    }

    // Export conversations
    exportConversations(format = 'json') {
        if (format === 'json') {
            return JSON.stringify(this.conversationHistory, null, 2);
        } else if (format === 'txt') {
            return this.conversationHistory.map(conv => 
                `[${new Date(conv.timestamp).toLocaleString()}]\nUser: ${conv.user}\nAI: ${conv.ai}\n\n`
            ).join('');
        }
        
        return null;
    }

    // Health check
    healthCheck() {
        const health = {
            status: 'healthy',
            components: {
                memory: this.memory ? 'ok' : 'error',
                nlp: this.nlp ? 'ok' : 'error',
                reasoning: this.reasoning ? 'ok' : 'error',
                brain: this.brain ? 'ok' : 'error'
            },
            uptime: Date.now(),
            timestamp: new Date().toISOString()
        };

        const hasErrors = Object.values(health.components).includes('error');
        if (hasErrors) {
            health.status = 'unhealthy';
        }

        return health;
    }
}

// Chat Interface cho browser
class ChatInterface {
    constructor(aiAgent) {
        this.ai = aiAgent;
        this.isTyping = false;
        this.initializeInterface();
    }

    initializeInterface() {
        // Create chat interface elements
        const chatContainer = document.getElementById('chat-container');
        if (!chatContainer) {
            console.error('Chat container not found');
            return;
        }

        this.addWelcomeMessage();
        this.setupEventListeners();
    }

    addWelcomeMessage() {
        this.addMessage(
            `Xin chào! Tôi là ${this.ai.name}. Tôi có thể học hỏi và trò chuyện với bạn về nhiều chủ đề khác nhau. Hãy hỏi tôi bất cứ điều gì!`,
            false
        );
    }

    setupEventListeners() {
        const input = document.getElementById('message-input');
        const sendButton = document.getElementById('send-button');

        if (input && sendButton) {
            input.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.sendMessage();
                }
            });

            sendButton.addEventListener('click', () => this.sendMessage());
        }
    }

    async sendMessage() {
        const input = document.getElementById('message-input');
        const message = input.value.trim();

        if (!message || this.isTyping) return;

        // Add user message
        this.addMessage(message, true);
        input.value = '';

        // Show typing indicator
        this.showTyping();

        try {
            // Get AI response
            const response = await this.ai.processMessage(message);
            
            // Hide typing and add AI response
            this.hideTyping();
            this.addMessage(response.content, false, {
                confidence: response.confidence,
                processingTime: response.processingTime,
                intent: response.intent?.intent,
                sentiment: response.sentiment?.sentiment
            });

        } catch (error) {
            this.hideTyping();
            this.addMessage('Xin lỗi, tôi gặp sự cố. Vui lòng thử lại.', false);
            console.error('Chat error:', error);
        }
    }

    addMessage(content, isUser, metadata = {}) {
        const messagesContainer = document.getElementById('messages');
        if (!messagesContainer) return;

        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isUser ? 'user' : 'ai'}`;
        
        const timestamp = new Date().toLocaleTimeString();
        let metadataInfo = '';
        
        if (!isUser && metadata.confidence) {
            metadataInfo = `<small class="metadata">Confidence: ${(metadata.confidence * 100).toFixed(0)}% | ${metadata.processingTime}ms</small>`;
        }

        messageDiv.innerHTML = `
            <div class="message-content">${content}</div>
            <div class="message-time">${timestamp}</div>
            ${metadataInfo}
        `;

        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    showTyping() {
        this.isTyping = true;
        const messagesContainer = document.getElementById('messages');
        
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message ai typing-indicator';
        typingDiv.id = 'typing-indicator';
        typingDiv.innerHTML = `
            <div class="message-content">
                <span class="typing-dots">
                    <span></span><span></span><span></span>
                </span>
                Đang suy nghĩ...
            </div>
        `;

        messagesContainer.appendChild(typingDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    hideTyping() {
        this.isTyping = false;
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }
}

// Main initialization
async function initializeAI() {
    console.log('🚀 Initializing AI Agent...');
    
    // Create AI Agent
    const agent = new AIAgent({
        name: 'AI Assistant',
        personality: 'helpful and friendly',
        language: 'vi',
        shortTermCapacity: 15,
        longTermCapacity: 2000,
        learning: true
    });

    // Create chat interface if in browser
    if (typeof window !== 'undefined') {
        window.aiAgent = agent;
        window.chatInterface = new ChatInterface(agent);
        
        // Expose useful functions globally
        window.saveAI = () => agent.save();
        window.loadAI = (data) => agent.load(data);
        window.getStats = () => agent.getStats();
        window.trainAI = () => agent.trainNetwork();
    }

    return agent;
}

// Export for Node.js
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        AIAgent,
        ChatInterface,
        initializeAI
    };
}

// Auto-initialize if in browser
if (typeof window !== 'undefined') {
    document.addEventListener('DOMContentLoaded', initializeAI);
}