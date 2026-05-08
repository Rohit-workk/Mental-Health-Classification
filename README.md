# 🚀 NLP Text Classification Project

## 📌 Project Overview

This project focuses on building an advanced **Text Classification System** using:

- Natural Language Processing (NLP)
- Machine Learning Algorithms
- Feature Engineering
- TF-IDF & Bag of Words
- Numerical Feature Analysis
- Hyperparameter Tuning
- LightGBM Optimization

The project compares multiple machine learning models using both:

- 📊 Bag of Words (BoW)
- 🧠 TF-IDF Vectorization

and finally develops an optimized classification pipeline.

---

# 📂 Project Workflow

```text
Raw Data
   ↓
Text Cleaning
   ↓
Feature Engineering
   ↓
Bag of Words / TF-IDF
   ↓
Numerical Feature Processing
   ↓
Model Training
   ↓
Hyperparameter Tuning
   ↓
LightGBM Optimization
   ↓
Final Pipeline
```

---

# 🧹 Text Preprocessing

Raw text data usually contains:

- Noise
- Special characters
- Stopwords
- Irregular formatting
- Unnecessary spaces

Preprocessing improves data quality and model learning.

---

## ✅ Preprocessing Steps

### 1️⃣ Lowercase Conversion

```python
text = text.lower()
```

---

### 2️⃣ Remove Special Characters

```python
import re

text = re.sub(r'[^a-zA-Z ]', '', text)
```

---

### 3️⃣ Tokenization

```python
from nltk.tokenize import word_tokenize

tokens = word_tokenize(text)
```

---

### 4️⃣ Stopword Removal

```python
from nltk.corpus import stopwords

tokens = [word for word in tokens if word not in stopwords.words('english')]
```

---

### 5️⃣ Stemming

```python
from nltk.stem import PorterStemmer

ps = PorterStemmer()
tokens = [ps.stem(word) for word in tokens]
```

---

# 🔢 Numerical Feature Engineering

Along with text data, several numerical features were created.

These features help models understand:

- Writing behavior
- Sentence structure
- Text complexity
- Word distribution

---

## 📌 Numerical Columns

```python
num_cols = [
    "num_words",
    "num_chars",
    "avg_word_length"
]
```

---

## 📊 Importance of Numerical Features

### ✅ num_words

Represents:
- Sentence length
- Writing pattern
- Content size

---

### ✅ num_chars

Represents:
- Character density
- Text size
- Complexity

---

### ✅ avg_word_length

Represents:
- Vocabulary richness
- Linguistic complexity
- Writing sophistication

---

# 📈 Log Transformation

The numerical columns were highly skewed and contained outliers.

To normalize the distribution, log transformation was applied.

---

## 📌 Formula

```text
x_new = log(1 + x)
```

---

## 📌 Implementation

```python
import numpy as np

for col in num_cols:
    df[col] = np.log1p(df[col])
```

---

## ✅ Benefits

- Reduces skewness
- Handles outliers
- Improves stability
- Improves convergence
- Better scaling

---

# 📝 Feature Extraction

Machine learning models cannot directly understand text.

Therefore text was converted into vectors using:

- Bag of Words
- TF-IDF Vectorizer

---

# 📌 Bag of Words (BoW)

Bag of Words converts text into word-frequency vectors.

---

## 📊 Bag of Words Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| MultinomialNB | 69.88% | 69.49% | 69.88% | 68.96% |
| BernoulliNB | 60.43% | 60.81% | 60.43% | 58.79% |
| LogisticRegression | 75.25% | 74.81% | 75.25% | 74.95% |
| DecisionTree | 66.84% | 66.74% | 66.84% | 66.78% |
| KNN | 54.42% | 55.64% | 54.42% | 49.29% |
| RandomForest | 69.65% | 74.61% | 69.65% | 68.27% |
| Bagging | 71.41% | 71.12% | 71.41% | 70.81% |
| AdaBoost | 67.29% | 66.60% | 67.29% | 66.04% |
| GradientBoosting | 72.53% | 72.82% | 72.53% | 71.99% |

---

# 📊 Analysis of Bag of Words

## ✅ Best Model

### Logistic Regression

- Accuracy: 75.25%
- F1-score: 74.95%

---

## 📌 Observations

### Logistic Regression
- Best overall performance
- Handles sparse vectors efficiently

### Gradient Boosting
- Strong ensemble performance

### KNN
- Weak performance on sparse NLP vectors

### Decision Tree
- Overfitting tendency

---

# 📌 TF-IDF Vectorization

TF-IDF improves text representation by reducing the importance of common words.

---

## 📊 TF-IDF Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| MultinomialNB | 65.96% | 71.12% | 65.96% | 63.24% |
| BernoulliNB | 60.85% | 62.14% | 60.85% | 59.27% |
| LogisticRegression | 76.14% | 76.03% | 76.14% | 75.70% |
| DecisionTree | 66.07% | 65.72% | 66.07% | 65.88% |
| KNN | 60.51% | 59.99% | 60.51% | 58.32% |
| RandomForest | 70.06% | 74.12% | 70.06% | 68.77% |
| Bagging | 71.34% | 71.08% | 71.34% | 70.72% |
| AdaBoost | 68.19% | 67.69% | 68.19% | 67.19% |
| GradientBoosting | 72.97% | 73.29% | 72.97% | 72.49% |

---

# 📊 TF-IDF Analysis

## 🏆 Best Model

### Logistic Regression

- Accuracy: 76.14%
- F1-score: 75.70%

---

## 🔥 Key Findings

### TF-IDF Outperformed Bag of Words

Reasons:
- Better feature quality
- Reduces influence of common words
- Better generalization

---

### Logistic Regression Performed Best

Reasons:
- Efficient for sparse matrices
- Strong linear decision boundaries
- High scalability

---

# ⚡ Hyperparameter Tuning

Hyperparameter tuning was performed using:

- GridSearchCV
- RandomizedSearchCV

---

# 📊 Balanced vs Non-Balanced Logistic Regression

| Metric | Balanced | Non-Balanced |
|---|---|---|
| Accuracy | 0.76 | 0.76 |
| Macro F1-score | 0.73 | 0.71 |
| Weighted F1-score | 0.76 | 0.76 |

---

# 🔥 Key Insights

## Minority Class Improvement

### Class 6
- Recall improved: 0.47 → 0.76
- F1 improved: 0.60 → 0.68

### Class 5
- Recall improved: 0.45 → 0.68
- F1 improved: 0.55 → 0.59

---

# ⚖️ Trade-off Analysis

## Balanced Model

### Advantages
- Better minority detection
- Higher fairness
- Better macro F1-score

### Disadvantages
- Slight precision reduction

---

## Non-Balanced Model

### Advantages
- Better majority class precision

### Disadvantages
- Poor minority detection
- Biased predictions

---

# 🏆 Final Model — LightGBM

After extensive experimentation, the final selected model was:

# ✅ LightGBM Classifier

---

## Why LightGBM?

- Faster training
- Better scalability
- Handles sparse TF-IDF matrices efficiently
- Strong ensemble learning
- Better generalization

---

# 📌 Final Pipeline

```python
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LGBMClassifier())
])
```

---

# 📦 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming |
| Pandas | Data Analysis |
| NumPy | Numerical Computing |
| Scikit-learn | Machine Learning |
| NLTK | NLP Preprocessing |
| LightGBM | Final Model |
| Flask/Streamlit | Deployment |

---

# 🚀 Future Improvements

- Deep Learning Models
- BERT / Transformers
- Explainable AI
- Model Monitoring
- Advanced Feature Engineering

---

# 👨‍💻 Author

## Rohit

B.Tech CSE (AI)

Interested in:
- Machine Learning
- NLP
- Deep Learning
- Data Science

---

# ⭐ Final Conclusion

This project demonstrates a complete NLP pipeline including:

✅ Text preprocessing  
✅ Feature engineering  
✅ TF-IDF analysis  
✅ Numerical feature integration  
✅ Model comparison  
✅ Hyperparameter tuning  
✅ LightGBM optimization  

The combination of:

- TF-IDF Features
- Numerical Features
- Balanced Learning
- LightGBM Optimization

helped achieve strong classification performance and better real-world applicability.
