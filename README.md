# Experimental CNN hyperparameter study with architecture comparison and performance analysis

A deep learning project focused on image classification using Convolutional Neural Networks (CNNs) on the CIFAR-10 dataset, with comparative experimentation on key hyperparameters such as stride size and convolution kernel size.

## Project Overview

This project explores the impact of CNN architectural hyperparameters on classification performance for the CIFAR-10 dataset.

The study includes:

- Building a baseline CNN model for image classification
- Performing hyperparameter experiments by changing:
  - Convolution stride
  - Filter (kernel) size
- Comparing train and test accuracy across multiple architectures
- Analyzing overfitting and underfitting behavior
- Building an improved optimized CNN architecture

The objective was to understand how CNN design choices affect performance on low-resolution image classification tasks.

---

## Dataset

**Dataset:** CIFAR-10

CIFAR-10 consists of 60,000 color images of size **32×32 pixels**, divided into 10 classes:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

Dataset split used:

- Training set: 70%
- Testing set: 30%

---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- PIL (Python Imaging Library)

---

## Project Workflow

### 1. Data Loading

- Loaded image labels from CSV
- Loaded image files from local dataset directory
- Converted images into NumPy arrays

### 2. Data Preprocessing

- Pixel normalization:

```python
X_train = X_train / 255.0
X_test = X_test / 255.0
```

- Label encoding using `LabelEncoder`

### 3. Baseline CNN Architecture

Baseline architecture:

```text
Input (32x32x3)
↓
Conv2D (32 filters, 3x3, stride=1)
↓
BatchNormalization
↓
MaxPooling
↓
Conv2D (64 filters, 3x3, stride=1)
↓
BatchNormalization
↓
MaxPooling
↓
Conv2D (128 filters, 3x3, stride=1)
↓
BatchNormalization
↓
MaxPooling
↓
Flatten
↓
Dense (256)
↓
Dropout
↓
Softmax Output (10 classes)
```

Loss Function:

```python
sparse_categorical_crossentropy
```

Optimizer:

```python
Adam
```

Callbacks used:

- EarlyStopping
- ModelCheckpoint
- ReduceLROnPlateau

Training:

```text
Epochs: 30
Batch Size: 64
```

---

## Hyperparameter Experiments

Four experiments were conducted to study the effect of stride and kernel size.

### Baseline Model

| Parameter | Value |
|---------|-------|
| Stride | 1 |
| Kernel Size | 3×3 |

Results:

- Train Accuracy: **83.64%**
- Test Accuracy: **80.17%**

---

### Case 1: Stride = 2, Kernel = 3×3

Results:

- Train Accuracy: **65.88%**
- Test Accuracy: **61.75%**

Observation:

Increasing stride reduced performance significantly due to aggressive spatial downsampling and information loss.

---

### Case 2: Stride = 3, Kernel = 3×3

Results:

- Train Accuracy: **61.73%**
- Test Accuracy: **58.43%**

Observation:

Further increasing stride caused severe underfitting as the CNN skipped too much image information.

---

### Case 3: Stride = 1, Kernel = 2×2

Results:

- Train Accuracy: **76.64%**
- Test Accuracy: **67.46%**

Observation:

Smaller kernels captured less contextual information, reducing classification performance.

---

### Case 4: Stride = 1, Kernel = 4×4

Results:

- Train Accuracy: **82.51%**
- Test Accuracy: **71.33%**

Observation:

Larger kernels improved context capture but increased model complexity and overfitting.

---

## Final Comparison

| Model | Stride | Kernel Size | Train Accuracy | Test Accuracy |
|------|--------|-------------|----------------|---------------|
| Baseline | 1 | 3×3 | 83.64% | 80.17% |
| Case 1 | 2 | 3×3 | 65.88% | 61.75% |
| Case 2 | 3 | 3×3 | 61.73% | 58.43% |
| Case 3 | 1 | 2×2 | 76.64% | 67.46% |
| Case 4 | 1 | 4×4 | 82.51% | 71.33% |

---

## Key Findings

### Effect of Stride

Increasing stride caused a consistent decrease in performance:

```text
Stride 1 → 80.17%
Stride 2 → 61.75%
Stride 3 → 58.43%
```

Reason:

- Larger stride skips pixels
- Reduced feature extraction quality
- Severe information loss in small 32×32 images

Conclusion:

**Smaller stride performs significantly better for CIFAR-10.**

---

### Effect of Kernel Size

Kernel size comparison:

```text
2×2 → 67.46%
3×3 → 80.17%
4×4 → 71.33%
```

Conclusion:

- 2×2 kernels captured insufficient context
- 4×4 kernels increased complexity and overfitting
- 3×3 provided the best balance

Final conclusion:

**3×3 convolution kernels performed best.**

---

## Improved CNN Model

An enhanced architecture was also implemented using:

- Data augmentation
- Additional convolution blocks
- GlobalAveragePooling
- AdamW optimizer
- Stronger regularization

Improved result:

- Final Test Accuracy: **81.03%**

This showed a modest improvement over the baseline.

---

## Model Performance Visualization

The project includes:

- Accuracy vs Epoch plots
- Loss vs Epoch plots
- Confusion matrix analysis
- Prediction probability visualization
- Sample image predictions

---

## Project Learnings

This project helped in understanding:

- CNN architecture design
- Role of stride in convolution
- Effect of filter size
- Feature extraction behavior
- Overfitting vs underfitting
- Model optimization techniques
- Hyperparameter experimentation
- Image preprocessing workflows

---

## Future Improvements

Possible future enhancements:

- ResNet implementation
- Transfer learning
- Better normalization strategies
- Label smoothing
- Larger/deeper CNN architectures
- Advanced data augmentation
- CIFAR-100 experimentation

---

## How to Run

Clone repository:

```bash
git clone https://github.com/your-username/cifar10-cnn-hyperparameter-study.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run notebook:

```bash
jupyter notebook
```

---

## Author

**Vivek Kumar Singh**
