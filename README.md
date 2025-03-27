# Suicide-Ideation-Detection-Experiments

### CNN-biLSTM-Attention (Text only) 
[View Notebook](Attention_experiment_same_weights.ipynb)  
Text only: Model weights are loaded from `init_weights.h5`  
Text and Personality : Model weights are loaded from `init_weights_2.h5`

Embedding Layer:
  - Embedding dimension: 300

Convolutional Layer:
  - 1D Convolution with 100 filters
  - Kernel size: 5
  - Activation function: ReLU

Pooling Layer:
  - MaxPooling1D

Bidirectional LSTM:
  - LSTM units: 100
  - Dropout: 0.3
  - Recurrent dropout: 0.3
  - Return sequences: True

LSTM Layer:
  - LSTM units: 100
  - Return sequences: True

Attention Mechanism:
  - Self-Attention Layer (SeqSelfAttention)
  - Activation function: Softmax

Fully Connected Layers:
  - Flatten Layer
  - Dense Layer: 64 units, ReLU activation
  - Dropout Layer: 0.3 dropout rate
  - Output Layer: Softmax activation

Training Hyperparameters:
  - Batch size: 64
  - Number of epochs: 5
  - Loss function: Categorical Crossentropyhttps://github.com/sylen111/Suicide-Ideation-Detection-Experiments/blob/main/README.md
  - Optimizer: Adam
  - Evaluation metric: Accuracy

### CNN-biLSTM-Attention (Text and Personality) 
[View Notebook](Attention_experiment_same_weights.ipynb)  

Embedding Layer:
  - Embedding dimension: 300

Convolutional Layer:
  - 1D Convolution with 100 filters
  - Kernel size: 5
  - Activation function: ReLU

Pooling Layer:
  - MaxPooling1D

Bidirectional LSTM:
  - LSTM units: *hyperparameter-optimized units*
  - Dropout: *hyperparameter-optimized rate*
  - Recurrent dropout: *hyperparameter-optimized rate*
  - Return sequences: True

Attention Mechanism:
  - Self-Attention Layer (SeqSelfAttention)
  - Activation function: Softmax

Fully Connected Layers:
  - Flatten Layer
  - Dense Layer: *hyperparameter-optimized units*, ReLU activation
  - Dropout Layer: *hyperparameter-optimized rate*
  - Output Layer: Softmax activation
    
Hyperparameter Optimization (Optuna Settings)
- 50 trials
- LSTM units: 100–150
- Dense units: 32–128
- Dropout rate: 0.1–0.6

Training Hyperparameters:
  - Batch size: 32
  - Number of epochs: 10
  - Loss function: Categorical Crossentropy
  - Optimizer: Adam
  - Evaluation metric: Accuracy


### HACNN-LSTM (Text only & Text and Personality traits)
[View Notebook](HACNN-LSTM.ipynb)  
Embedding Layer:
- Embedding dimension: 300

Convolutional Layer:
- 1D Convolution with 100 filters
- Kernel size: 5
- Activation function: ReLU

Pooling Layer:
- MaxPooling1D

LSTM Layer:
- LSTM units: 100
- Return sequences: True

Attention Mechanism:
- Multi-Head Attention Layer
- Number of heads: 5
- Key dimension: 64

Fully Connected Layers:
- Flatten Layer
- Dense Layer: 64 units, ReLU activation
- Dropout Layer: 0.3 dropout rate
- Output Layer: Softmax activation 

Training Hyperparameters:
- Batch size: 32
- Loss function: Categorical Crossentropy

PSO Optimization Settings:
- Number of particles: 10
- Optimization steps: 50
- Acceleration: 1.0
- Local rate: 0.6
- Global rate: 0.4
