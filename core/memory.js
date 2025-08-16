/**
 * 🧠 Memory & Reasoning System
 * Hệ thống ghi nhớ và suy luận cho AI Agent
 */

class Memory {
    constructor(config = {}) {
        this.shortTermCapacity = config.shortTermCapacity || 10;
        this.longTermCapacity = config.longTermCapacity || 1000;
        this.episodicCapacity = config.episodicCapacity || 500;
        
        // Different types of memory
        this.shortTerm = []; // Cuộc trò chuyện hiện tại
        this.longTerm = new Map(); // Kiến thức lâu dài
        this.episodic = []; // Kỷ niệm các sự kiện
        this.semantic = new Map(); // Kiến thức về thế giới
        this.procedural = new Map(); // Cách thực hiện các tác vụ
        
        // Memory metadata
        this.memoryWeights = new Map(); // Trọng số quan trọng
        this.accessCount = new Map(); // Số lần truy cập
        this.lastAccess = new Map(); // Lần truy cập cuối
        
        this.decayRate = 0.1; // Tốc độ quên
        this.consolidationThreshold = 5; // Ngưỡng củng cố
    }

    // Thêm vào bộ nhớ ngắn hạn
    addToShortTerm(item) {
        const memoryItem = {
            content: item,
            timestamp: Date.now(),
            importance: this.calculateImportance(item),
            id: this.generateId()
        };
        
        this.shortTerm.push(memoryItem);
        
        // Giới hạn dung lượng
        if (this.shortTerm.length > this.shortTermCapacity) {
            const removed = this.shortTerm.shift();
            // Có thể chuyển sang long-term nếu quan trọng
            if (removed.importance > 0.7) {
                this.consolidateToLongTerm(removed);
            }
        }
        
        return memoryItem.id;
    }

    // Thêm vào bộ nhớ dài hạn
    addToLongTerm(key, value, importance = 0.5) {
        const memoryItem = {
            content: value,
            importance,
            timestamp: Date.now(),
            accessCount: 0,
            lastAccess: Date.now(),
            consolidated: false
        };
        
        this.longTerm.set(key, memoryItem);
        this.memoryWeights.set(key, importance);
        
        // Quản lý dung lượng
        this.manageLongTermCapacity();
        
        return key;
    }

    // Thêm kỷ niệm sự kiện
    addEpisodicMemory(event) {
        const episode = {
            event,
            timestamp: Date.now(),
            context: this.getCurrentContext(),
            emotions: this.analyzeEmotionalContext(event),
            importance: this.calculateImportance(event),
            id: this.generateId()
        };
        
        this.episodic.push(episode);
        
        if (this.episodic.length > this.episodicCapacity) {
            this.episodic.shift();
        }
        
        return episode.id;
    }

    // Truy xuất từ bộ nhớ
    recall(query, memoryType = 'all') {
        const results = [];
        
        if (memoryType === 'all' || memoryType === 'shortterm') {
            results.push(...this.searchShortTerm(query));
        }
        
        if (memoryType === 'all' || memoryType === 'longterm') {
            results.push(...this.searchLongTerm(query));
        }
        
        if (memoryType === 'all' || memoryType === 'episodic') {
            results.push(...this.searchEpisodic(query));
        }
        
        // Sắp xếp theo độ liên quan và cập nhật access
        return results
            .sort((a, b) => b.relevance - a.relevance)
            .map(item => {
                this.updateAccess(item.id);
                return item;
            });
    }

    // Tìm kiếm trong bộ nhớ ngắn hạn
    searchShortTerm(query) {
        return this.shortTerm
            .map(item => ({
                ...item,
                relevance: this.calculateRelevance(query, item.content),
                type: 'shortterm'
            }))
            .filter(item => item.relevance > 0.1);
    }

    // Tìm kiếm trong bộ nhớ dài hạn
    searchLongTerm(query) {
        const results = [];
        
        for (const [key, item] of this.longTerm.entries()) {
            const relevance = this.calculateRelevance(query, item.content);
            if (relevance > 0.1) {
                results.push({
                    id: key,
                    ...item,
                    relevance,
                    type: 'longterm'
                });
            }
        }
        
        return results;
    }

    // Tìm kiếm trong bộ nhớ kỷ niệm
    searchEpisodic(query) {
        return this.episodic
            .map(episode => ({
                ...episode,
                relevance: this.calculateRelevance(query, episode.event),
                type: 'episodic'
            }))
            .filter(episode => episode.relevance > 0.1);
    }

    // Tính toán độ quan trọng
    calculateImportance(content) {
        let importance = 0.5;
        
        if (typeof content === 'string') {
            // Keywords quan trọng
            const importantKeywords = ['quan trọng', 'cần nhớ', 'important', 'remember', 'critical'];
            const hasImportantKeywords = importantKeywords.some(keyword => 
                content.toLowerCase().includes(keyword));
            
            if (hasImportantKeywords) importance += 0.3;
            
            // Độ dài và phức tạp
            if (content.length > 100) importance += 0.1;
            
            // Cảm xúc mạnh
            const emotionalWords = ['yêu', 'ghét', 'tuyệt vời', 'tệ hại', 'love', 'hate'];
            const hasEmotions = emotionalWords.some(word => 
                content.toLowerCase().includes(word));
            
            if (hasEmotions) importance += 0.2;
        }
        
        return Math.min(importance, 1.0);
    }

    // Tính toán độ liên quan
    calculateRelevance(query, content) {
        if (typeof query !== 'string' || typeof content !== 'string') {
            return 0;
        }
        
        const queryTokens = query.toLowerCase().split(' ');
        const contentTokens = content.toLowerCase().split(' ');
        
        let matches = 0;
        let totalTokens = queryTokens.length;
        
        queryTokens.forEach(token => {
            if (contentTokens.some(cToken => 
                cToken.includes(token) || token.includes(cToken))) {
                matches++;
            }
        });
        
        // Similarity score
        const similarity = matches / totalTokens;
        
        // Boost for exact matches
        if (content.toLowerCase().includes(query.toLowerCase())) {
            return Math.min(similarity + 0.5, 1.0);
        }
        
        return similarity;
    }

    // Củng cố vào bộ nhớ dài hạn
    consolidateToLongTerm(shortTermItem) {
        const key = `consolidated_${shortTermItem.id}`;
        this.addToLongTerm(key, shortTermItem.content, shortTermItem.importance);
        
        // Đánh dấu là đã được củng cố
        if (this.longTerm.has(key)) {
            this.longTerm.get(key).consolidated = true;
        }
    }

    // Quản lý dung lượng bộ nhớ dài hạn
    manageLongTermCapacity() {
        if (this.longTerm.size <= this.longTermCapacity) {
            return;
        }
        
        // Lấy các items ít quan trọng nhất
        const itemsByImportance = Array.from(this.longTerm.entries())
            .map(([key, item]) => ({ key, ...item }))
            .sort((a, b) => {
                // Sắp xếp theo importance, access count và thời gian
                const importanceScore = a.importance - b.importance;
                const accessScore = a.accessCount - b.accessCount;
                const timeScore = a.lastAccess - b.lastAccess;
                
                return importanceScore + accessScore * 0.1 + timeScore * 0.001;
            });
        
        // Xóa items ít quan trọng
        const toRemove = itemsByImportance.slice(0, this.longTerm.size - this.longTermCapacity);
        toRemove.forEach(({ key }) => {
            this.longTerm.delete(key);
            this.memoryWeights.delete(key);
            this.accessCount.delete(key);
            this.lastAccess.delete(key);
        });
    }

    // Cập nhật thông tin truy cập
    updateAccess(id) {
        const count = this.accessCount.get(id) || 0;
        this.accessCount.set(id, count + 1);
        this.lastAccess.set(id, Date.now());
        
        // Tăng importance nếu được truy cập nhiều
        if (count > this.consolidationThreshold) {
            const currentWeight = this.memoryWeights.get(id) || 0.5;
            this.memoryWeights.set(id, Math.min(currentWeight + 0.1, 1.0));
        }
    }

    // Memory decay - quên theo thời gian
    applyDecay() {
        const now = Date.now();
        const decayThreshold = 24 * 60 * 60 * 1000; // 24 hours
        
        // Decay long-term memory
        for (const [key, item] of this.longTerm.entries()) {
            const timeSinceAccess = now - item.lastAccess;
            if (timeSinceAccess > decayThreshold) {
                const decayFactor = Math.exp(-this.decayRate * (timeSinceAccess / decayThreshold));
                item.importance *= decayFactor;
                
                // Remove if too unimportant
                if (item.importance < 0.1 && !item.consolidated) {
                    this.longTerm.delete(key);
                    this.memoryWeights.delete(key);
                    this.accessCount.delete(key);
                    this.lastAccess.delete(key);
                }
            }
        }
    }

    // Lấy context hiện tại
    getCurrentContext() {
        return {
            recentMemories: this.shortTerm.slice(-3),
            timestamp: Date.now(),
            activeTopics: this.getActiveTopics()
        };
    }

    // Phân tích context cảm xúc
    analyzeEmotionalContext(content) {
        // Đơn giản hóa - có thể tích hợp với sentiment analysis
        const emotionalKeywords = {
            happy: ['vui', 'hạnh phúc', 'vui vẻ', 'happy', 'joy', 'excited'],
            sad: ['buồn', 'khóc', 'sad', 'cry', 'depressed'],
            angry: ['tức giận', 'tức', 'angry', 'mad', 'furious'],
            surprised: ['ngạc nhiên', 'surprised', 'amazed', 'shocked']
        };
        
        const emotions = {};
        for (const [emotion, keywords] of Object.entries(emotionalKeywords)) {
            emotions[emotion] = keywords.some(keyword => 
                content.toLowerCase().includes(keyword)) ? 1 : 0;
        }
        
        return emotions;
    }

    // Lấy các chủ đề đang hoạt động
    getActiveTopics() {
        const topics = [];
        const recentContent = this.shortTerm
            .slice(-5)
            .map(item => item.content)
            .join(' ');
        
        // Extract topics from recent content
        const topicKeywords = {
            technology: ['công nghệ', 'ai', 'robot', 'computer', 'technology'],
            health: ['sức khỏe', 'bệnh', 'thuốc', 'health', 'medicine'],
            education: ['học', 'giáo dục', 'trường', 'education', 'school'],
            entertainment: ['phim', 'nhạc', 'game', 'movie', 'music']
        };
        
        for (const [topic, keywords] of Object.entries(topicKeywords)) {
            if (keywords.some(keyword => recentContent.toLowerCase().includes(keyword))) {
                topics.push(topic);
            }
        }
        
        return topics;
    }

    // Generate unique ID
    generateId() {
        return Date.now().toString(36) + Math.random().toString(36).substr(2);
    }

    // Lưu memory state
    save() {
        return {
            shortTerm: this.shortTerm,
            longTerm: Array.from(this.longTerm.entries()),
            episodic: this.episodic,
            semantic: Array.from(this.semantic.entries()),
            procedural: Array.from(this.procedural.entries()),
            memoryWeights: Array.from(this.memoryWeights.entries()),
            accessCount: Array.from(this.accessCount.entries()),
            lastAccess: Array.from(this.lastAccess.entries()),
            config: {
                shortTermCapacity: this.shortTermCapacity,
                longTermCapacity: this.longTermCapacity,
                episodicCapacity: this.episodicCapacity,
                decayRate: this.decayRate,
                consolidationThreshold: this.consolidationThreshold
            },
            timestamp: Date.now()
        };
    }

    // Load memory state
    load(data) {
        this.shortTerm = data.shortTerm || [];
        this.longTerm = new Map(data.longTerm || []);
        this.episodic = data.episodic || [];
        this.semantic = new Map(data.semantic || []);
        this.procedural = new Map(data.procedural || []);
        this.memoryWeights = new Map(data.memoryWeights || []);
        this.accessCount = new Map(data.accessCount || []);
        this.lastAccess = new Map(data.lastAccess || []);
        
        if (data.config) {
            Object.assign(this, data.config);
        }
        
        console.log('✅ Memory system loaded successfully!');
    }

    // Thống kê memory
    getStats() {
        return {
            shortTerm: {
                count: this.shortTerm.length,
                capacity: this.shortTermCapacity,
                usage: (this.shortTerm.length / this.shortTermCapacity * 100).toFixed(1) + '%'
            },
            longTerm: {
                count: this.longTerm.size,
                capacity: this.longTermCapacity,
                usage: (this.longTerm.size / this.longTermCapacity * 100).toFixed(1) + '%'
            },
            episodic: {
                count: this.episodic.length,
                capacity: this.episodicCapacity,
                usage: (this.episodic.length / this.episodicCapacity * 100).toFixed(1) + '%'
            },
            totalMemories: this.shortTerm.length + this.longTerm.size + this.episodic.length,
            mostAccessed: this.getMostAccessedMemories(5),
            activeTopics: this.getActiveTopics()
        };
    }

    // Lấy memories được truy cập nhiều nhất
    getMostAccessedMemories(limit = 5) {
        return Array.from(this.accessCount.entries())
            .sort(([,a], [,b]) => b - a)
            .slice(0, limit)
            .map(([id, count]) => ({
                id,
                accessCount: count,
                lastAccess: this.lastAccess.get(id),
                content: this.longTerm.get(id)?.content || 'N/A'
            }));
    }
}

/**
 * 🤔 Reasoning Engine
 * Hệ thống suy luận logic cho AI Agent
 */
class ReasoningEngine {
    constructor(memory) {
        this.memory = memory;
        this.rules = new Map();
        this.facts = new Set();
        this.inferences = new Map();
        this.reasoningHistory = [];
        
        // Load basic reasoning rules
        this.initializeBasicRules();
    }

    // Khởi tạo các quy tắc suy luận cơ bản
    initializeBasicRules() {
        // Logical rules
        this.addRule('modus_ponens', {
            pattern: ['if {A} then {B}', '{A}'],
            conclusion: '{B}',
            confidence: 0.9
        });

        this.addRule('modus_tollens', {
            pattern: ['if {A} then {B}', 'not {B}'],
            conclusion: 'not {A}',
            confidence: 0.85
        });

        // Causal reasoning
        this.addRule('causal_chain', {
            pattern: ['{A} causes {B}', '{B} causes {C}'],
            conclusion: '{A} may cause {C}',
            confidence: 0.7
        });

        // Analogical reasoning
        this.addRule('analogy', {
            pattern: ['{A} is like {B}', '{A} has property {P}'],
            conclusion: '{B} may have property {P}',
            confidence: 0.6
        });

        // Temporal reasoning
        this.addRule('temporal_sequence', {
            pattern: ['{A} happens before {B}', '{B} is happening'],
            conclusion: '{A} has happened',
            confidence: 0.8
        });
    }

    // Thêm quy tắc suy luận
    addRule(name, rule) {
        this.rules.set(name, {
            ...rule,
            name,
            createdAt: Date.now(),
            usageCount: 0
        });
    }

    // Thêm fact
    addFact(fact, confidence = 1.0) {
        const factObj = {
            content: fact,
            confidence,
            timestamp: Date.now(),
            source: 'user_input'
        };
        
        this.facts.add(JSON.stringify(factObj));
        return factObj;
    }

    // Suy luận từ facts và rules
    reason(query, maxDepth = 3) {
        console.log(`🤔 Đang suy luận về: "${query}"`);
        
        const reasoningChain = {
            query,
            startTime: Date.now(),
            steps: [],
            conclusions: [],
            confidence: 0
        };

        // Direct fact lookup
        const directFacts = this.findRelevantFacts(query);
        if (directFacts.length > 0) {
            reasoningChain.steps.push({
                type: 'direct_fact',
                content: directFacts[0],
                confidence: directFacts[0].confidence
            });
            reasoningChain.conclusions.push(directFacts[0].content);
            reasoningChain.confidence = directFacts[0].confidence;
        }

        // Memory-based reasoning
        const memoryResults = this.memory.recall(query);
        if (memoryResults.length > 0) {
            const bestMemory = memoryResults[0];
            reasoningChain.steps.push({
                type: 'memory_recall',
                content: bestMemory.content,
                relevance: bestMemory.relevance,
                confidence: bestMemory.importance
            });
            
            if (reasoningChain.conclusions.length === 0) {
                reasoningChain.conclusions.push(bestMemory.content);
                reasoningChain.confidence = bestMemory.relevance * bestMemory.importance;
            }
        }

        // Rule-based reasoning
        this.applyRules(query, reasoningChain, 0, maxDepth);

        // Pattern matching
        this.patternMatching(query, reasoningChain);

        // Analogical reasoning
        this.analogicalReasoning(query, reasoningChain);

        reasoningChain.endTime = Date.now();
        reasoningChain.duration = reasoningChain.endTime - reasoningChain.startTime;

        // Store reasoning history
        this.reasoningHistory.push(reasoningChain);
        if (this.reasoningHistory.length > 100) {
            this.reasoningHistory.shift();
        }

        return reasoningChain;
    }

    // Áp dụng rules
    applyRules(query, reasoningChain, depth, maxDepth) {
        if (depth >= maxDepth) return;

        for (const [ruleName, rule] of this.rules.entries()) {
            if (this.matchesRule(query, rule)) {
                const conclusion = this.applyRule(query, rule);
                if (conclusion) {
                    reasoningChain.steps.push({
                        type: 'rule_application',
                        rule: ruleName,
                        conclusion: conclusion.content,
                        confidence: conclusion.confidence,
                        depth
                    });

                    if (!reasoningChain.conclusions.includes(conclusion.content)) {
                        reasoningChain.conclusions.push(conclusion.content);
                        reasoningChain.confidence = Math.max(
                            reasoningChain.confidence, 
                            conclusion.confidence
                        );
                    }

                    // Recursive reasoning
                    this.applyRules(conclusion.content, reasoningChain, depth + 1, maxDepth);
                }
            }
        }
    }

    // Kiểm tra rule matching
    matchesRule(query, rule) {
        // Simplified pattern matching
        return rule.pattern.some(pattern => {
            const regex = pattern.replace(/\{[^}]+\}/g, '(.+)');
            return new RegExp(regex, 'i').test(query);
        });
    }

    // Áp dụng một rule cụ thể
    applyRule(input, rule) {
        try {
            // Extract variables from pattern
            const variables = this.extractVariables(input, rule.pattern[0]);
            
            if (variables && Object.keys(variables).length > 0) {
                let conclusion = rule.conclusion;
                
                // Substitute variables
                for (const [varName, varValue] of Object.entries(variables)) {
                    conclusion = conclusion.replace(`{${varName}}`, varValue);
                }

                rule.usageCount++;
                
                return {
                    content: conclusion,
                    confidence: rule.confidence,
                    rule: rule.name
                };
            }
        } catch (error) {
            console.warn(`Error applying rule ${rule.name}:`, error);
        }
        
        return null;
    }

    // Trích xuất variables từ pattern
    extractVariables(text, pattern) {
        const variables = {};
        const regex = pattern.replace(/\{([^}]+)\}/g, (match, varName) => {
            return `(?<${varName}>.+?)`;
        });

        const match = text.match(new RegExp(regex, 'i'));
        if (match && match.groups) {
            return match.groups;
        }

        return null;
    }

    // Pattern matching
    patternMatching(query, reasoningChain) {
        const patterns = [
            {
                pattern: /what is (.+)/i,
                handler: (match) => this.explainConcept(match[1])
            },
            {
                pattern: /how to (.+)/i,
                handler: (match) => this.provideInstructions(match[1])
            },
            {
                pattern: /why (.+)/i,
                handler: (match) => this.explainReason(match[1])
            },
            {
                pattern: /when (.+)/i,
                handler: (match) => this.provideTimeInfo(match[1])
            }
        ];

        for (const { pattern, handler } of patterns) {
            const match = query.match(pattern);
            if (match) {
                const result = handler(match);
                if (result) {
                    reasoningChain.steps.push({
                        type: 'pattern_matching',
                        pattern: pattern.source,
                        result: result.content,
                        confidence: result.confidence
                    });
                    
                    if (!reasoningChain.conclusions.includes(result.content)) {
                        reasoningChain.conclusions.push(result.content);
                        reasoningChain.confidence = Math.max(
                            reasoningChain.confidence,
                            result.confidence
                        );
                    }
                }
                break;
            }
        }
    }

    // Analogical reasoning
    analogicalReasoning(query, reasoningChain) {
        const similarMemories = this.memory.recall(query, 'episodic')
            .filter(memory => memory.relevance > 0.5)
            .slice(0, 3);

        for (const memory of similarMemories) {
            const analogy = this.findAnalogy(query, memory.content);
            if (analogy) {
                reasoningChain.steps.push({
                    type: 'analogical_reasoning',
                    source: memory.content,
                    analogy: analogy.content,
                    confidence: analogy.confidence
                });

                if (!reasoningChain.conclusions.some(c => c.includes(analogy.content))) {
                    reasoningChain.conclusions.push(`Based on analogy: ${analogy.content}`);
                    reasoningChain.confidence = Math.max(
                        reasoningChain.confidence,
                        analogy.confidence * 0.7 // Lower confidence for analogies
                    );
                }
            }
        }
    }

    // Tìm analogy
    findAnalogy(source, target) {
        // Simplified analogy detection
        const sourceTokens = source.toLowerCase().split(' ');
        const targetTokens = target.toLowerCase().split(' ');
        
        const commonTokens = sourceTokens.filter(token => 
            targetTokens.includes(token) && token.length > 3
        );

        if (commonTokens.length > 0) {
            return {
                content: `Similar to: ${target}`,
                confidence: commonTokens.length / Math.max(sourceTokens.length, targetTokens.length)
            };
        }

        return null;
    }

    // Explain concept
    explainConcept(concept) {
        const conceptMemories = this.memory.recall(concept);
        if (conceptMemories.length > 0) {
            return {
                content: `${concept} is ${conceptMemories[0].content}`,
                confidence: conceptMemories[0].relevance
            };
        }
        
        return {
            content: `I need more information to explain ${concept}`,
            confidence: 0.3
        };
    }

    // Provide instructions
    provideInstructions(task) {
        const instructionMemories = this.memory.recall(`how to ${task}`);
        if (instructionMemories.length > 0) {
            return {
                content: instructionMemories[0].content,
                confidence: instructionMemories[0].relevance
            };
        }

        return {
            content: `To ${task}, you might need to break it down into smaller steps`,
            confidence: 0.4
        };
    }

    // Explain reason
    explainReason(statement) {
        const causalMemories = this.memory.recall(`because ${statement}`);
        if (causalMemories.length > 0) {
            return {
                content: causalMemories[0].content,
                confidence: causalMemories[0].relevance
            };
        }

        return {
            content: `The reason for ${statement} might be related to cause and effect`,
            confidence: 0.3
        };
    }

    // Provide time info
    provideTimeInfo(event) {
        const timeMemories = this.memory.recall(`when ${event}`);
        if (timeMemories.length > 0) {
            return {
                content: timeMemories[0].content,
                confidence: timeMemories[0].relevance
            };
        }

        return {
            content: `I don't have specific timing information about ${event}`,
            confidence: 0.2
        };
    }

    // Tìm relevant facts
    findRelevantFacts(query) {
        const relevantFacts = [];
        
        for (const factStr of this.facts) {
            const fact = JSON.parse(factStr);
            if (fact.content.toLowerCase().includes(query.toLowerCase()) ||
                query.toLowerCase().includes(fact.content.toLowerCase())) {
                relevantFacts.push(fact);
            }
        }

        return relevantFacts.sort((a, b) => b.confidence - a.confidence);
    }

    // Lưu reasoning state
    save() {
        return {
            rules: Array.from(this.rules.entries()),
            facts: Array.from(this.facts),
            inferences: Array.from(this.inferences.entries()),
            reasoningHistory: this.reasoningHistory.slice(-50), // Keep last 50
            timestamp: Date.now()
        };
    }

    // Load reasoning state
    load(data) {
        this.rules = new Map(data.rules || []);
        this.facts = new Set(data.facts || []);
        this.inferences = new Map(data.inferences || []);
        this.reasoningHistory = data.reasoningHistory || [];
        
        console.log('✅ Reasoning engine loaded successfully!');
    }

    // Thống kê reasoning
    getStats() {
        return {
            totalRules: this.rules.size,
            totalFacts: this.facts.size,
            totalInferences: this.inferences.size,
            reasoningHistoryLength: this.reasoningHistory.length,
            mostUsedRules: Array.from(this.rules.entries())
                .sort(([,a], [,b]) => b.usageCount - a.usageCount)
                .slice(0, 5)
                .map(([name, rule]) => ({ name, usageCount: rule.usageCount })),
            averageReasoningTime: this.calculateAverageReasoningTime()
        };
    }

    // Tính thời gian suy luận trung bình
    calculateAverageReasoningTime() {
        if (this.reasoningHistory.length === 0) return 0;
        
        const totalTime = this.reasoningHistory.reduce((sum, reasoning) => 
            sum + (reasoning.duration || 0), 0);
        
        return (totalTime / this.reasoningHistory.length).toFixed(2) + 'ms';
    }
}

// Export classes
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        Memory,
        ReasoningEngine
    };
} else {
    // Browser environment
    window.Memory = Memory;
    window.ReasoningEngine = ReasoningEngine;
}