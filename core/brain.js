/**
 * 🧠 Neural Network Brain - Core AI Engine
 * Tự implement neural network không dùng thư viện ngoài
 */

class Matrix {
    constructor(rows, cols, data = null) {
        this.rows = rows;
        this.cols = cols;
        this.data = data || Array(rows).fill().map(() => Array(cols).fill(0));
    }

    static random(rows, cols) {
        const matrix = new Matrix(rows, cols);
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                matrix.data[i][j] = Math.random() * 2 - 1; // -1 to 1
            }
        }
        return matrix;
    }

    multiply(other) {
        if (this.cols !== other.rows) {
            throw new Error('Matrix dimensions incompatible for multiplication');
        }
        
        const result = new Matrix(this.rows, other.cols);
        for (let i = 0; i < result.rows; i++) {
            for (let j = 0; j < result.cols; j++) {
                let sum = 0;
                for (let k = 0; k < this.cols; k++) {
                    sum += this.data[i][k] * other.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        return result;
    }

    add(other) {
        const result = new Matrix(this.rows, this.cols);
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                result.data[i][j] = this.data[i][j] + other.data[i][j];
            }
        }
        return result;
    }

    subtract(other) {
        const result = new Matrix(this.rows, this.cols);
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                result.data[i][j] = this.data[i][j] - other.data[i][j];
            }
        }
        return result;
    }

    transpose() {
        const result = new Matrix(this.cols, this.rows);
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                result.data[j][i] = this.data[i][j];
            }
        }
        return result;
    }

    map(func) {
        const result = new Matrix(this.rows, this.cols);
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                result.data[i][j] = func(this.data[i][j]);
            }
        }
        return result;
    }
}

class ActivationFunctions {
    static relu(x) {
        return Math.max(0, x);
    }

    static reluDerivative(x) {
        return x > 0 ? 1 : 0;
    }

    static sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }

    static sigmoidDerivative(x) {
        const s = ActivationFunctions.sigmoid(x);
        return s * (1 - s);
    }

    static tanh(x) {
        return Math.tanh(x);
    }

    static tanhDerivative(x) {
        const t = Math.tanh(x);
        return 1 - t * t;
    }

    static softmax(matrix) {
        const result = new Matrix(matrix.rows, matrix.cols);
        for (let i = 0; i < matrix.rows; i++) {
            let sum = 0;
            const maxVal = Math.max(...matrix.data[i]);
            
            // Tính exp và sum
            for (let j = 0; j < matrix.cols; j++) {
                result.data[i][j] = Math.exp(matrix.data[i][j] - maxVal);
                sum += result.data[i][j];
            }
            
            // Normalize
            for (let j = 0; j < matrix.cols; j++) {
                result.data[i][j] /= sum;
            }
        }
        return result;
    }
}

class NeuralLayer {
    constructor(inputSize, outputSize, activation = 'relu') {
        this.weights = Matrix.random(inputSize, outputSize);
        this.biases = Matrix.random(1, outputSize);
        this.activation = activation;
        this.lastInput = null;
        this.lastOutput = null;
        this.lastZ = null;
    }

    forward(input) {
        this.lastInput = input;
        
        // Z = X * W + B
        this.lastZ = input.multiply(this.weights).add(this.biases);
        
        // Apply activation function
        let output;
        switch (this.activation) {
            case 'relu':
                output = this.lastZ.map(ActivationFunctions.relu);
                break;
            case 'sigmoid':
                output = this.lastZ.map(ActivationFunctions.sigmoid);
                break;
            case 'tanh':
                output = this.lastZ.map(ActivationFunctions.tanh);
                break;
            case 'softmax':
                output = ActivationFunctions.softmax(this.lastZ);
                break;
            default:
                output = this.lastZ;
        }
        
        this.lastOutput = output;
        return output;
    }

    backward(gradOutput, learningRate = 0.001) {
        // Gradient of activation function
        let gradZ;
        switch (this.activation) {
            case 'relu':
                gradZ = this.lastZ.map(ActivationFunctions.reluDerivative);
                break;
            case 'sigmoid':
                gradZ = this.lastZ.map(ActivationFunctions.sigmoidDerivative);
                break;
            case 'tanh':
                gradZ = this.lastZ.map(ActivationFunctions.tanhDerivative);
                break;
            default:
                gradZ = new Matrix(this.lastZ.rows, this.lastZ.cols, 
                    Array(this.lastZ.rows).fill().map(() => Array(this.lastZ.cols).fill(1)));
        }

        // Element-wise multiplication
        const gradZActivation = new Matrix(gradOutput.rows, gradOutput.cols);
        for (let i = 0; i < gradOutput.rows; i++) {
            for (let j = 0; j < gradOutput.cols; j++) {
                gradZActivation.data[i][j] = gradOutput.data[i][j] * gradZ.data[i][j];
            }
        }

        // Gradients for weights and biases
        const gradWeights = this.lastInput.transpose().multiply(gradZActivation);
        const gradBiases = gradZActivation;

        // Gradient for input (to pass to previous layer)
        const gradInput = gradZActivation.multiply(this.weights.transpose());

        // Update weights and biases
        const weightUpdate = gradWeights.map(x => x * learningRate);
        const biasUpdate = gradBiases.map(x => x * learningRate);

        this.weights = this.weights.subtract(weightUpdate);
        this.biases = this.biases.subtract(biasUpdate);

        return gradInput;
    }
}

class NeuralNetwork {
    constructor(config = {}) {
        this.config = {
            layers: config.layers || [100, 50, 25],
            activations: config.activations || ['relu', 'relu', 'sigmoid'],
            learningRate: config.learningRate || 0.001,
            ...config
        };
        
        this.layers = [];
        this.initializeLayers();
        this.trainingHistory = [];
    }

    initializeLayers() {
        for (let i = 0; i < this.config.layers.length - 1; i++) {
            const layer = new NeuralLayer(
                this.config.layers[i],
                this.config.layers[i + 1],
                this.config.activations[i] || 'relu'
            );
            this.layers.push(layer);
        }
    }

    forward(input) {
        let output = input;
        for (const layer of this.layers) {
            output = layer.forward(output);
        }
        return output;
    }

    backward(gradOutput) {
        let grad = gradOutput;
        for (let i = this.layers.length - 1; i >= 0; i--) {
            grad = this.layers[i].backward(grad, this.config.learningRate);
        }
    }

    train(inputs, targets, epochs = 1000) {
        console.log('🧠 Bắt đầu training neural network...');
        
        for (let epoch = 0; epoch < epochs; epoch++) {
            let totalLoss = 0;
            
            for (let i = 0; i < inputs.length; i++) {
                // Forward pass
                const input = new Matrix(1, inputs[i].length, [inputs[i]]);
                const target = new Matrix(1, targets[i].length, [targets[i]]);
                
                const output = this.forward(input);
                
                // Calculate loss (Mean Squared Error)
                const loss = this.calculateLoss(output, target);
                totalLoss += loss;
                
                // Backward pass
                const gradOutput = this.calculateGradient(output, target);
                this.backward(gradOutput);
            }
            
            const avgLoss = totalLoss / inputs.length;
            
            if (epoch % 100 === 0) {
                console.log(`Epoch ${epoch}, Loss: ${avgLoss.toFixed(4)}`);
            }
            
            this.trainingHistory.push({
                epoch,
                loss: avgLoss,
                timestamp: Date.now()
            });
        }
        
        console.log('✅ Training hoàn thành!');
    }

    calculateLoss(output, target) {
        let loss = 0;
        for (let i = 0; i < output.rows; i++) {
            for (let j = 0; j < output.cols; j++) {
                const diff = output.data[i][j] - target.data[i][j];
                loss += diff * diff;
            }
        }
        return loss / (output.rows * output.cols);
    }

    calculateGradient(output, target) {
        const gradient = new Matrix(output.rows, output.cols);
        for (let i = 0; i < output.rows; i++) {
            for (let j = 0; j < output.cols; j++) {
                gradient.data[i][j] = 2 * (output.data[i][j] - target.data[i][j]);
            }
        }
        return gradient;
    }

    predict(input) {
        const inputMatrix = new Matrix(1, input.length, [input]);
        const output = this.forward(inputMatrix);
        return output.data[0];
    }

    save() {
        const modelData = {
            config: this.config,
            weights: this.layers.map(layer => ({
                weights: layer.weights.data,
                biases: layer.biases.data,
                activation: layer.activation
            })),
            trainingHistory: this.trainingHistory,
            timestamp: Date.now()
        };
        return JSON.stringify(modelData);
    }

    load(modelData) {
        const data = typeof modelData === 'string' ? JSON.parse(modelData) : modelData;
        
        this.config = data.config;
        this.trainingHistory = data.trainingHistory || [];
        
        this.layers = [];
        for (let i = 0; i < data.weights.length; i++) {
            const layerData = data.weights[i];
            const layer = new NeuralLayer(
                layerData.weights.length,
                layerData.weights[0].length,
                layerData.activation
            );
            
            layer.weights.data = layerData.weights;
            layer.biases.data = layerData.biases;
            
            this.layers.push(layer);
        }
        
        console.log('✅ Model đã được load thành công!');
    }

    // Thêm regularization để tránh overfitting
    addRegularization(lambda = 0.01) {
        for (const layer of this.layers) {
            // L2 regularization
            for (let i = 0; i < layer.weights.rows; i++) {
                for (let j = 0; j < layer.weights.cols; j++) {
                    layer.weights.data[i][j] *= (1 - lambda);
                }
            }
        }
    }

    // Evaluate model performance
    evaluate(testInputs, testTargets) {
        let correct = 0;
        let totalLoss = 0;
        
        for (let i = 0; i < testInputs.length; i++) {
            const prediction = this.predict(testInputs[i]);
            const target = testTargets[i];
            
            // Calculate accuracy (for classification)
            const predictedClass = prediction.indexOf(Math.max(...prediction));
            const actualClass = target.indexOf(Math.max(...target));
            
            if (predictedClass === actualClass) {
                correct++;
            }
            
            // Calculate loss
            let loss = 0;
            for (let j = 0; j < prediction.length; j++) {
                const diff = prediction[j] - target[j];
                loss += diff * diff;
            }
            totalLoss += loss / prediction.length;
        }
        
        const accuracy = correct / testInputs.length;
        const avgLoss = totalLoss / testInputs.length;
        
        return {
            accuracy: accuracy * 100,
            loss: avgLoss,
            correct,
            total: testInputs.length
        };
    }
}

// Export classes
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        NeuralNetwork,
        Matrix,
        ActivationFunctions,
        NeuralLayer
    };
} else {
    // Browser environment
    window.NeuralNetwork = NeuralNetwork;
    window.Matrix = Matrix;
    window.ActivationFunctions = ActivationFunctions;
    window.NeuralLayer = NeuralLayer;
}