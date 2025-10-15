# Deep Learning Project: CNN (Fashion-MNIST) and BiLSTM (SST-2)

This repository contains two end-to-end notebook workflows:

- A Convolutional Neural Network (CNN) for Fashion-MNIST image classification
- A Bidirectional LSTM (BiLSTM) for SST-2 sentiment analysis

---

## Dataset details

### Fashion-MNIST (for CNN)

- 70,000 grayscale images of clothing items (60,000 train / 10,000 test)
- Image size: 28×28 pixels, 10 classes:
  - `T-shirt/top`, `Trouser`, `Pullover`, `Dress`, `Coat`, `Sandal`, `Shirt`, `Sneaker`, `Bag`, `Ankle boot`
- Loading: `tensorflow.keras.datasets.fashion_mnist.load_data()`
- Preprocessing:
  - Normalize pixel values to [0, 1]
  - Expand channel dimension to shape `(28, 28, 1)` for CNN input
  - One-hot encode labels to 10 classes

### GLUE SST-2 (for BiLSTM)

- Stanford Sentiment Treebank binary classification (positive/negative)
- Loaded via `tensorflow_datasets` as `glue/sst2`
- Splits used in the notebook:
  - Train: `split='train'`
  - Validation/Test: `split='validation'`
- Preprocessing:
  - Keras `Tokenizer` with `num_words=10000` and OOV token
  - Convert texts to integer sequences
  - Pad/truncate sequences to `max_length=50` (post-padding)
  - Labels cast to `float32` for training

---

## Model architectures

### CNN (Fashion-MNIST)

A compact architecture suitable for 28×28 grayscale images:

- Input: `(28, 28, 1)`
- Conv2D(32, 3×3, ReLU)
- MaxPooling2D(2×2)
- Conv2D(64, 3×3, ReLU)
- MaxPooling2D(2×2)
- Flatten
- Dense(64, ReLU)
- Dropout(0.5)
- Dense(10, Softmax)

Compile and training settings:

- Optimizer: `adam`
- Loss: `categorical_crossentropy`
- Metric: `accuracy`
- Epochs: `10`
- Batch size: `32`
- Validation: uses the provided test split

### BiLSTM (SST-2 Sentiment)

A text classifier with embeddings and stacked bidirectional LSTMs:

- Input: integer sequences of length 50, vocabulary size 10,000
- Embedding(`input_dim=10000`, `output_dim=256`, `input_length=50`)
- SpatialDropout1D(0.2)
- Bidirectional(LSTM(128, `return_sequences=True`))
- Dropout(0.3)
- Bidirectional(LSTM(64))
- Dropout(0.3)
- Dense(64, ReLU)
- Dropout(0.5)
- Dense(1, Sigmoid)

Compile and training settings:

- Optimizer: `adam`
- Loss: `binary_crossentropy`
- Metrics: `accuracy`
- Epochs: `5`
- Batch size: `32`
- Validation split: `0.2` from training set
- Callbacks:
  - EarlyStopping(monitor=`val_loss`, patience=5, restore_best_weights=True)
  - ReduceLROnPlateau(monitor=`val_loss`, factor=0.5, patience=3, min_lr=1e-7)

---

## Evaluation, Metrics, and Outputs

### CNN on Fashion-MNIST

**Dataset:** Fashion-MNIST (60,000 training images, 10,000 test images)  
**Model Type:** Convolutional Neural Network (2 Conv2D + 2 MaxPool + Dense layers)  
**Loss Function:** Categorical Cross-Entropy  
**Optimizer:** Adam  

#### Metrics Summary

| Metric | Training | Validation | Test |
|---------|-----------|-------------|------|
| Accuracy | 91.3% | 90.6% | **90.64%** |
| Loss | 0.238 | 0.265 | **0.2654** |

#### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|------------|---------|-----------|----------|
| T-shirt/Top | 0.87 | 0.84 | 0.85 | 1000 |
| Trouser | 0.99 | 0.98 | 0.99 | 1000 |
| Pullover | 0.82 | 0.87 | 0.85 | 1000 |
| Dress | 0.90 | 0.92 | 0.91 | 1000 |
| Coat | 0.84 | 0.88 | 0.86 | 1000 |
| Sandal | 0.98 | 0.98 | 0.98 | 1000 |
| Shirt | 0.75 | 0.69 | 0.72 | 1000 |
| Sneaker | 0.95 | 0.97 | 0.96 | 1000 |
| Bag | 0.99 | 0.97 | 0.98 | 1000 |
| Ankle Boot | 0.98 | 0.96 | 0.97 | 1000 |

**Macro Avg:** 0.91 **Weighted Avg:** 0.91 **Overall Accuracy:** **0.9064**

#### Visual Outputs

- **Training Curves:**  
  - Accuracy improved steadily to ~91% across 10 epochs.  
  - Validation loss decreased and stabilized around 0.27.  
- **Confusion Matrix:** Highlights strong classification in classes like *Trouser*, *Sandal*, and *Ankle Boot*, with minor confusion among garment categories such as *Shirt* and *Coat*.

### BiLSTM for Sentiment Analysis

**Dataset:** GLUE SST-2 (Stanford Sentiment Treebank)  
**Model Type:** Bidirectional LSTM with Embedding and Dropout  
**Loss Function:** Binary Cross-Entropy  
**Optimizer:** Adam  
**Callbacks:** EarlyStopping, ReduceLROnPlateau  

#### Metrics Summary

| Metric | Training | Validation | Test |
|---------|-----------|-------------|------|
| Accuracy | 96.8% | 92.0% | **82.91%** |
| Loss | 0.074 | 0.338 | **0.436** |

#### Confusion Matrix

|              | Predicted Negative | Predicted Positive |
|---------------|--------------------|--------------------|
| **Actual Negative** | 340 | 88 |
| **Actual Positive** | 61 | 383 |

**Precision:** 0.81 **Recall:** 0.86 **F1 Score:** 0.84  

#### Visual Outputs

- **Training Curves:**  
  - Training accuracy increased to 96% by epoch 5, with validation convergence around 92%.  
  - Overfitting mitigated by dropout and early stopping.  
- **Confusion Matrix Plot:**  
  Shows good discrimination between positive and negative sentiment classes, minor false positives.  
