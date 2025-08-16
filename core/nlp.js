/**
 * 🗣️ Natural Language Processing Engine
 * Xử lý ngôn ngữ tự nhiên cho tiếng Việt và English
 */

class TextProcessor {
    constructor() {
        this.vietnameseStopWords = new Set([
            'là', 'của', 'và', 'có', 'trong', 'một', 'với', 'được', 'để', 'các',
            'cho', 'từ', 'này', 'đó', 'những', 'người', 'khi', 'về', 'như',
            'tôi', 'bạn', 'chúng', 'họ', 'nó', 'mình', 'ta', 'anh', 'chị'
        ]);
        
        this.englishStopWords = new Set([
            'the', 'is', 'at', 'which', 'on', 'a', 'an', 'as', 'are', 'was',
            'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'should', 'could', 'can', 'may', 'might', 'must'
        ]);

        this.vocabulary = new Map();
        this.wordFrequency = new Map();
        this.bigramFrequency = new Map();
        this.trigramFrequency = new Map();
    }

    // Tokenization - Tách từ
    tokenize(text) {
        // Xử lý tiếng Việt có dấu
        const cleanText = text
            .toLowerCase()
            .replace(/[^\w\sáàảãạăắằẳẵặâấầẩẫậđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ]/g, ' ')
            .replace(/\s+/g, ' ')
            .trim();

        return cleanText.split(' ').filter(token => token.length > 0);
    }

    // Remove stop words
    removeStopWords(tokens, language = 'auto') {
        const stopWords = language === 'vi' ? this.vietnameseStopWords : 
                         language === 'en' ? this.englishStopWords :
                         new Set([...this.vietnameseStopWords, ...this.englishStopWords]);
        
        return tokens.filter(token => !stopWords.has(token));
    }

    // Detect language
    detectLanguage(text) {
        const vietnameseChars = /[áàảãạăắằẳẵặâấầẩẫậđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ]/;
        return vietnameseChars.test(text) ? 'vi' : 'en';
    }

    // Build vocabulary from corpus
    buildVocabulary(texts) {
        console.log('📚 Đang xây dựng từ điển...');
        
        texts.forEach(text => {
            const tokens = this.tokenize(text);
            const language = this.detectLanguage(text);
            const cleanTokens = this.removeStopWords(tokens, language);
            
            // Unigrams
            cleanTokens.forEach(token => {
                if (!this.vocabulary.has(token)) {
                    this.vocabulary.set(token, this.vocabulary.size);
                }
                
                const count = this.wordFrequency.get(token) || 0;
                this.wordFrequency.set(token, count + 1);
            });
            
            // Bigrams
            for (let i = 0; i < cleanTokens.length - 1; i++) {
                const bigram = `${cleanTokens[i]} ${cleanTokens[i + 1]}`;
                const count = this.bigramFrequency.get(bigram) || 0;
                this.bigramFrequency.set(bigram, count + 1);
            }
            
            // Trigrams
            for (let i = 0; i < cleanTokens.length - 2; i++) {
                const trigram = `${cleanTokens[i]} ${cleanTokens[i + 1]} ${cleanTokens[i + 2]}`;
                const count = this.trigramFrequency.get(trigram) || 0;
                this.trigramFrequency.set(trigram, count + 1);
            }
        });
        
        console.log(`✅ Từ điển đã xây dựng: ${this.vocabulary.size} từ`);
    }

    // Text to vector conversion
    textToVector(text, maxLength = 100) {
        const tokens = this.tokenize(text);
        const language = this.detectLanguage(text);
        const cleanTokens = this.removeStopWords(tokens, language);
        
        const vector = new Array(Math.min(maxLength, this.vocabulary.size)).fill(0);
        
        cleanTokens.forEach(token => {
            const index = this.vocabulary.get(token);
            if (index !== undefined && index < vector.length) {
                vector[index] = 1; // Binary encoding hoặc có thể dùng TF-IDF
            }
        });
        
        return vector;
    }

    // TF-IDF calculation
    calculateTFIDF(text, corpus) {
        const tokens = this.removeStopWords(this.tokenize(text));
        const tfidf = {};
        
        // Term Frequency
        const termFreq = {};
        tokens.forEach(token => {
            termFreq[token] = (termFreq[token] || 0) + 1;
        });
        
        // Document Frequency
        const docFreq = {};
        corpus.forEach(doc => {
            const docTokens = new Set(this.removeStopWords(this.tokenize(doc)));
            docTokens.forEach(token => {
                docFreq[token] = (docFreq[token] || 0) + 1;
            });
        });
        
        // Calculate TF-IDF
        Object.keys(termFreq).forEach(token => {
            const tf = termFreq[token] / tokens.length;
            const idf = Math.log(corpus.length / (docFreq[token] || 1));
            tfidf[token] = tf * idf;
        });
        
        return tfidf;
    }

    // Named Entity Recognition - Nhận dạng thực thể
    recognizeEntities(text) {
        const entities = [];
        const tokens = this.tokenize(text);
        
        // Patterns for different entity types
        const patterns = {
            person: /^(anh|chị|ông|bà|thầy|cô|em|chú|cậu)\s+([A-Za-zÀ-ỹ]+)/i,
            location: /^(ở|tại|đến|từ)\s+([A-Za-zÀ-ỹ\s]+)/i,
            organization: /^(công ty|trường|đại học|viện)\s+([A-Za-zÀ-ỹ\s]+)/i,
            time: /\d{1,2}:\d{2}|\d{1,2}\/\d{1,2}\/\d{2,4}|hôm nay|ngày mai|hôm qua/i,
            number: /\d+/g,
            email: /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/g,
            phone: /\b\d{10,11}\b/g
        };
        
        Object.entries(patterns).forEach(([type, pattern]) => {
            const matches = text.match(pattern);
            if (matches) {
                matches.forEach(match => {
                    entities.push({
                        type,
                        value: match.trim(),
                        position: text.indexOf(match)
                    });
                });
            }
        });
        
        return entities;
    }

    // Part-of-Speech Tagging - Gán nhãn từ loại
    posTagging(tokens) {
        const posRules = {
            // Danh từ
            noun: ['người', 'việc', 'vật', 'thứ', 'cái', 'con', 'bản', 'cuộc'],
            // Động từ  
            verb: ['là', 'có', 'được', 'làm', 'đi', 'đến', 'về', 'nói', 'biết', 'thấy'],
            // Tính từ
            adjective: ['tốt', 'xấu', 'đẹp', 'to', 'nhỏ', 'cao', 'thấp', 'nhanh', 'chậm'],
            // Trạng từ
            adverb: ['rất', 'khá', 'hơi', 'cực', 'siêu', 'quá', 'lắm', 'nhiều', 'ít'],
            // Giới từ
            preposition: ['trong', 'ngoài', 'trên', 'dưới', 'về', 'cho', 'với', 'từ', 'đến']
        };
        
        return tokens.map(token => {
            for (const [pos, words] of Object.entries(posRules)) {
                if (words.includes(token.toLowerCase())) {
                    return { word: token, pos };
                }
            }
            
            // Default rules based on patterns
            if (/\d+/.test(token)) return { word: token, pos: 'number' };
            if (token.length === 1) return { word: token, pos: 'particle' };
            
            return { word: token, pos: 'unknown' };
        });
    }

    // Sentiment Analysis - Phân tích cảm xúc
    analyzeSentiment(text) {
        const positiveWords = new Set([
            'tốt', 'hay', 'đẹp', 'tuyệt', 'xuất sắc', 'hoàn hảo', 'yêu', 'thích', 
            'vui', 'hạnh phúc', 'tích cực', 'good', 'great', 'excellent', 'amazing',
            'love', 'like', 'happy', 'positive', 'wonderful', 'fantastic'
        ]);
        
        const negativeWords = new Set([
            'xấu', 'tệ', 'dở', 'ghét', 'không thích', 'buồn', 'tức giận', 'tiêu cực',
            'bad', 'terrible', 'awful', 'hate', 'dislike', 'sad', 'angry', 'negative'
        ]);
        
        const tokens = this.removeStopWords(this.tokenize(text));
        let score = 0;
        let positiveCount = 0;
        let negativeCount = 0;
        
        tokens.forEach(token => {
            if (positiveWords.has(token)) {
                score += 1;
                positiveCount++;
            } else if (negativeWords.has(token)) {
                score -= 1;
                negativeCount++;
            }
        });
        
        const sentiment = score > 0 ? 'positive' : score < 0 ? 'negative' : 'neutral';
        const confidence = Math.abs(score) / Math.max(tokens.length, 1);
        
        return {
            sentiment,
            score,
            confidence,
            positiveCount,
            negativeCount,
            totalWords: tokens.length
        };
    }

    // Intent Classification - Phân loại ý định
    classifyIntent(text) {
        const intents = {
            greeting: ['xin chào', 'chào', 'hello', 'hi', 'hey', 'good morning', 'good evening'],
            question: ['gì', 'ai', 'đâu', 'khi nào', 'tại sao', 'như thế nào', 'what', 'who', 'where', 'when', 'why', 'how'],
            request: ['làm ơn', 'xin hãy', 'có thể', 'giúp', 'please', 'can you', 'could you', 'help me'],
            goodbye: ['tạm biệt', 'chào tạm biệt', 'bye', 'goodbye', 'see you later'],
            compliment: ['tuyệt vời', 'giỏi', 'thông minh', 'great job', 'well done', 'excellent'],
            complaint: ['không hài lòng', 'tệ', 'không tốt', 'terrible', 'awful', 'disappointed']
        };
        
        const tokens = this.tokenize(text.toLowerCase());
        const scores = {};
        
        Object.entries(intents).forEach(([intent, keywords]) => {
            let score = 0;
            keywords.forEach(keyword => {
                if (tokens.some(token => token.includes(keyword) || keyword.includes(token))) {
                    score += 1;
                }
            });
            scores[intent] = score;
        });
        
        const maxScore = Math.max(...Object.values(scores));
        const predictedIntent = Object.keys(scores).find(intent => scores[intent] === maxScore);
        
        return {
            intent: maxScore > 0 ? predictedIntent : 'unknown',
            confidence: maxScore / Math.max(tokens.length, 1),
            scores
        };
    }

    // Extract keywords - Trích xuất từ khóa
    extractKeywords(text, topN = 5) {
        const tokens = this.removeStopWords(this.tokenize(text));
        const frequency = {};
        
        tokens.forEach(token => {
            if (token.length > 2) { // Bỏ qua từ quá ngắn
                frequency[token] = (frequency[token] || 0) + 1;
            }
        });
        
        const keywords = Object.entries(frequency)
            .sort(([,a], [,b]) => b - a)
            .slice(0, topN)
            .map(([word, freq]) => ({ word, frequency: freq }));
        
        return keywords;
    }

    // Text similarity using Jaccard coefficient
    calculateSimilarity(text1, text2) {
        const tokens1 = new Set(this.removeStopWords(this.tokenize(text1)));
        const tokens2 = new Set(this.removeStopWords(this.tokenize(text2)));
        
        const intersection = new Set([...tokens1].filter(x => tokens2.has(x)));
        const union = new Set([...tokens1, ...tokens2]);
        
        return intersection.size / union.size;
    }

    // Generate n-grams
    generateNGrams(tokens, n = 2) {
        const ngrams = [];
        for (let i = 0; i <= tokens.length - n; i++) {
            ngrams.push(tokens.slice(i, i + n).join(' '));
        }
        return ngrams;
    }

    // Text preprocessing pipeline
    preprocess(text, options = {}) {
        const {
            lowercase = true,
            removeStopwords = true,
            detectLanguage = true,
            extractEntities = false,
            analyzeSentiment = false,
            classifyIntent = false,
            extractKeywords = false
        } = options;
        
        let processedText = text;
        if (lowercase) {
            processedText = processedText.toLowerCase();
        }
        
        const tokens = this.tokenize(processedText);
        const language = detectLanguage ? this.detectLanguage(text) : null;
        const cleanTokens = removeStopwords ? this.removeStopWords(tokens, language) : tokens;
        
        const result = {
            originalText: text,
            processedText,
            tokens: cleanTokens,
            language
        };
        
        if (extractEntities) {
            result.entities = this.recognizeEntities(text);
        }
        
        if (analyzeSentiment) {
            result.sentiment = this.analyzeSentiment(text);
        }
        
        if (classifyIntent) {
            result.intent = this.classifyIntent(text);
        }
        
        if (extractKeywords) {
            result.keywords = this.extractKeywords(text);
        }
        
        return result;
    }

    // Save model data
    save() {
        return {
            vocabulary: Array.from(this.vocabulary.entries()),
            wordFrequency: Array.from(this.wordFrequency.entries()),
            bigramFrequency: Array.from(this.bigramFrequency.entries()),
            trigramFrequency: Array.from(this.trigramFrequency.entries()),
            timestamp: Date.now()
        };
    }

    // Load model data
    load(data) {
        this.vocabulary = new Map(data.vocabulary);
        this.wordFrequency = new Map(data.wordFrequency);
        this.bigramFrequency = new Map(data.bigramFrequency);
        this.trigramFrequency = new Map(data.trigramFrequency);
        
        console.log('✅ NLP model loaded successfully!');
    }
}

// Language Model for text generation
class LanguageModel {
    constructor(nlpProcessor) {
        this.nlp = nlpProcessor;
        this.ngramModel = new Map();
        this.contextWindow = 3;
    }

    // Build n-gram language model
    buildModel(texts) {
        console.log('🔤 Đang xây dựng language model...');
        
        texts.forEach(text => {
            const tokens = ['<START>', ...this.nlp.tokenize(text), '<END>'];
            
            for (let i = 0; i < tokens.length - this.contextWindow; i++) {
                const context = tokens.slice(i, i + this.contextWindow).join(' ');
                const nextWord = tokens[i + this.contextWindow];
                
                if (!this.ngramModel.has(context)) {
                    this.ngramModel.set(context, new Map());
                }
                
                const contextMap = this.ngramModel.get(context);
                contextMap.set(nextWord, (contextMap.get(nextWord) || 0) + 1);
            }
        });
        
        console.log('✅ Language model hoàn thành!');
    }

    // Generate text based on context
    generateText(seedText, maxLength = 50) {
        const tokens = ['<START>', ...this.nlp.tokenize(seedText)];
        const generated = [...tokens];
        
        for (let i = 0; i < maxLength; i++) {
            const context = generated.slice(-this.contextWindow).join(' ');
            const possibleNext = this.ngramModel.get(context);
            
            if (!possibleNext || possibleNext.size === 0) {
                break;
            }
            
            // Weighted random selection
            const choices = Array.from(possibleNext.entries());
            const totalWeight = choices.reduce((sum, [, weight]) => sum + weight, 0);
            
            let random = Math.random() * totalWeight;
            let selectedWord = '<END>';
            
            for (const [word, weight] of choices) {
                random -= weight;
                if (random <= 0) {
                    selectedWord = word;
                    break;
                }
            }
            
            if (selectedWord === '<END>') {
                break;
            }
            
            generated.push(selectedWord);
        }
        
        return generated.slice(tokens.length).join(' ');
    }
}

// Export classes
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        TextProcessor,
        LanguageModel
    };
} else {
    // Browser environment
    window.TextProcessor = TextProcessor;
    window.LanguageModel = LanguageModel;
}