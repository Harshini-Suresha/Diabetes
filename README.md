# Exploratory Data Analysis & Classification on Diabetes Dataset  
### *(Multi-Trial Comparative Study — Trials 1, 2 & 3)*

---

## Project Overview

This repository contains three experimental trials (`Trial_1`, `Trial_2`, and `Trial_3`) of **Exploratory Data Analysis (EDA)** and **K-Nearest Neighbors (KNN) Classification** performed on the **Diabetes Dataset** from Kaggle.  

Each trial explores different preprocessing, visualization, and modeling strategies to study the effect of feature scaling, outlier handling, and model parameter tuning on diabetes prediction performance.

All experiments are available in both **Jupyter Notebook (.ipynb)** and **Google Colab-compatible (.xpynb)** formats for reproducibility.

---

## Objective

The main goal is to build an interpretable and efficient **classification pipeline** that predicts whether a patient is likely to have diabetes based on diagnostic attributes such as:
- Glucose concentration
- Blood Pressure
- BMI
- Age
- Insulin levels
- Diabetes Pedigree Function, etc.

The trials focus on:
- Data understanding and cleaning
- Visualization-driven insights
- Preprocessing (scaling, outlier handling, NaN treatment)
- Classification using **K-Nearest Neighbors (KNN)**
- Comparative performance evaluation across trials


---

## 📚 Dataset Information

**Source:** [Akshay Dattatray Khare — Diabetes Dataset (Kaggle)](https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset)

**Instances:** 768  
**Attributes:** 9 (8 features + 1 outcome)  

| Feature | Description |
|----------|--------------|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-Hour serum insulin (mu U/ml) |
| BMI | Body Mass Index (kg/m²) |
| DiabetesPedigreeFunction | Diabetes genetic likelihood |
| Age | Age in years |
| Outcome | Binary outcome (1: Diabetic, 0: Non-diabetic) |

---

## Experimental Trials Summary

### **Trial 1 — Baseline EDA & KNN**
**Goal:** Establish a clean baseline with minimal preprocessing.

- Loaded dataset directly from Kaggle.
- Conducted basic EDA (`.head()`, `.describe()`, `.info()`).
- Visualized distributions (histograms, countplots).
- Used raw features for KNN classification without scaling.
- Evaluated model performance using accuracy and confusion matrix.

**Outcome:**  
- Accuracy ≈ **69–71%** on test data.  
- Model performance limited due to lack of feature scaling.

---

### **Trial 2 — Enhanced Preprocessing & Correlation Analysis**
**Goal:** Improve accuracy by refining feature preparation.

- Handled infinite and missing values using replacement and imputation.
- Applied **StandardScaler** for normalization of all features.
- Generated detailed **pairplots** and **correlation heatmaps** to identify key predictors.
- Explored outlier detection using **boxplots** and **z-score thresholding**.
- Fine-tuned **KNN hyperparameter (k)** in range (1–15).

**Outcome:**  
- Optimal k = **3**  
- Test accuracy improved to **74–76%**  
- Notable correlation: `Glucose`, `BMI`, and `Age` with Outcome.

---

### **Trial 3 — Optimized Scaling, Validation, and Visualization**
**Goal:** Implement robust validation and refined visualization-driven insights.

- Removed extreme outliers and performed data balancing check.
- Applied both **MinMaxScaler** and **StandardScaler** for comparative analysis.
- Conducted EDA using **Seaborn** and **Matplotlib** with multi-variable visualizations.
- Used **train-test split (70-30)** and evaluated over multiple random seeds.
- Added **classification report** (precision, recall, F1-score).
- Visualized KNN performance curves (train vs test accuracy over different k values).

**Outcome:**  
- Optimal k = **1**  
- **Train accuracy:** 100%  
- **Test accuracy:** 75.18%  
- **Balanced precision/recall**, strong overall generalization.

---

## Visualization Highlights

- **Outcome Distribution:** Class balance between diabetic and non-diabetic patients.  
- **Boxplots:** Outlier identification in `Insulin` and `SkinThickness`.  
- **Histograms:** Feature distributions across population.  
- **Pairplot:** Feature interaction colored by diabetes outcome.  
- **Heatmap:** Correlation among continuous variables.  
- **Accuracy Curve:** Optimal K selection visualized.

---

## Model Workflow

### 1. Import Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
```
---

### 2. Load and Explore Dataset
```python
data = pd.read_csv("/kaggle/input/diabetes-dataset/diabetes.csv")
data.info()
data.describe()
```
---


### 3. Preprocess and Scale
```python
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(data.drop('Outcome', axis=1)), columns=data.columns[:-1])
y = data['Outcome']
```
---

### 4. Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```
---

###5. Train KNN and Tune K
```python
test_scores, train_scores = [], []
for k in range(1,15):
    knn = KNeighborsClassifier(k)
    knn.fit(X_train, y_train)
    train_scores.append(knn.score(X_train, y_train))
    test_scores.append(knn.score(X_test, y_test))
```
---

### 6. Evaluate
```python
knn_best = KNeighborsClassifier(1)
knn_best.fit(X_train, y_train)
y_pred = knn_best.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

## 📊 Results Comparison Across Trials

| Trial | Preprocessing Method              | Optimal K | Train Accuracy | Test Accuracy | Notes                 |
|:------|:----------------------------------|:----------|:----------------|:---------------|:----------------------|
| **Trial 1** | None (Raw Data)                 | 5 | 88.5% | 70.1% | Baseline experiment |
| **Trial 2** | StandardScaler + Outlier Filtering | 3 | 94.8% | 74.6% | Balanced improvement |
| **Trial 3** | Advanced Scaling + Validation     | 1 | 100%  | 75.2% | Final optimized model |


---

Install dependencies:
```python
pip install -r requirements.txt
```
