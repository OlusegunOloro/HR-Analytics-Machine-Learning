# HR Analytics Machine Learning 📊

> **Fairness-Aware Employee Promotion Prediction using Machine Learning**

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-blue)
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-green)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-red)
![Responsible AI](https://img.shields.io/badge/Responsible%20AI-Fairness%20Aware-success)

---

## Executive Summary

Employee promotion decisions have significant implications for talent retention, workforce development, and organisational diversity. Traditional promotion processes may unintentionally introduce bias, leading to unequal opportunities across demographic groups.

This project develops an end-to-end machine learning pipeline to predict employee promotions while investigating fairness across gender groups. A **Decision Tree Classifier** was trained on a cleaned and preprocessed HR dataset, achieving **95.19% test accuracy** and **94.67% cross-validation accuracy** following hyperparameter optimisation using **GridSearchCV with 10-fold cross-validation**.

Beyond predictive performance, this work incorporates fairness evaluation through group-wise confusion matrices, demographic parity, equal opportunity, and accuracy comparisons between protected groups.

The project demonstrates how machine learning can support human resource decision-making while promoting transparency, accountability, and responsible AI principles.

---

## Business Problem

Employee promotion processes are often subjective and susceptible to unconscious bias. Organisations require fair and data-driven systems capable of identifying high-performing employees without disadvantaging protected groups.

The objective of this project is to:

- Predict employee promotion outcomes using historical HR data.
- Investigate whether model predictions differ across gender groups.
- Evaluate fairness metrics alongside traditional machine learning performance measures.
- Demonstrate responsible AI practices within Human Resource Analytics.

---

## Project Objectives

- Build an end-to-end machine learning workflow for promotion prediction.
- Perform exploratory data analysis to identify key promotion factors.
- Clean and preprocess HR data for robust model training.
- Handle missing values using appropriate imputation strategies.
- Optimise model performance through hyperparameter tuning.
- Evaluate model effectiveness using multiple classification metrics.
- Assess fairness using gender as a protected attribute.
- Provide actionable insights for responsible HR decision-making.

---

## Dataset Overview

The dataset contains employee demographic, training, and performance information.

| Feature | Description |
|----------|-------------|
| department | Department of employee |
| region | Region of employment |
| education | Educational qualification |
| gender | Protected attribute used for fairness evaluation |
| recruitment_channel | Recruitment source |
| no_of_trainings | Number of completed trainings |
| age | Employee age |
| previous_year_rating | Performance rating from previous year |
| length_of_service | Years with organisation |
| awards_won | Number of awards received |
| avg_training_score | Average training assessment score |
| promoted | Target variable |

### Target Variable

text
0 = Not Promoted
1 = Promoted


---

# Exploratory Data Analysis (EDA)

The analysis explored:

- Promotion distribution across departments.
- Gender representation.
- Training performance patterns.
- Service length distributions.
- Previous performance ratings.
- Awards and promotion relationships.
- Missing value patterns.
- Outlier detection.

Key observations included:

- Employees with missing `previous_year_rating` values typically had only one year of service.
- Award recipients showed higher promotion probabilities.
- Previous performance ratings strongly influenced promotion outcomes.

---

# Data Cleaning & Preprocessing

## Column Standardisation

The dataset contained inconsistent feature names which were standardised:

python
'awards_won?'  → 'awards_won'
'is_promoted' → 'promoted'


The `employee_id` feature was removed because it provided no predictive value.

---

## Missing Value Analysis

| Feature | Missing Values | Percentage |
|----------|----------------|------------|
| education | 2,409 | 4.40% |
| previous_year_rating | 4,124 | 7.52% |

---

## Imputation Strategy

Different strategies were selected according to feature characteristics.

| Feature | Method | Reason |
|----------|----------|----------|
| previous_year_rating | Median Imputation | Robust against outliers |
| education | Most Frequent Imputation | Preserves category distribution |

Implementation:

python
from sklearn.impute import SimpleImputer

rating_imputer = SimpleImputer(strategy='median')
education_imputer = SimpleImputer(strategy='most_frequent')


After preprocessing:

✅ No missing values remained.

---

# Machine Learning Methodology

## Algorithm

The project utilised a:

### 🌳 Decision Tree Classifier

Decision Trees were selected because they:

- Are highly interpretable.
- Handle categorical and numerical variables effectively.
- Capture nonlinear relationships.
- Provide transparent decision-making processes.
- Align well with responsible AI requirements.

---

# Hyperparameter Optimisation

Hyperparameter tuning was performed using:

python
GridSearchCV(
    estimator=DecisionTreeClassifier(),
    cv=10
)


## Search Space

python
param_grid = {
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [4, 6, 8, 10],
    'min_samples_leaf': [2, 4, 6, 8],
    'criterion': ['gini', 'entropy']
}


---

## Best Hyperparameters

| Parameter | Value |
|------------|--------|
| Criterion | Gini |
| Max Depth | 20 |
| Min Samples Leaf | 2 |
| Min Samples Split | 4 |
| Cross Validation | 10-Fold |

---

## Cross Validation Performance

| Metric | Score |
|----------|--------|
| Best CV Accuracy | **94.67%** |

---

# Model Performance

## Training Performance

| Metric | Score |
|----------|--------|
| Accuracy | 98.22% |
| Precision | 96.66% |
| Recall | 99.91% |

---

## Test Performance

| Metric | Score |
|----------|--------|
| Accuracy | **95.19%** |
| Precision | **91.37%** |
| Recall | **99.69%** |

The relatively small performance gap between training and test datasets indicates good generalisation capability with limited overfitting.

---

# Fairness Evaluation

Responsible AI principles formed a core component of this project.

The model was evaluated independently across gender groups using:

- Group-specific confusion matrices
- Classification reports
- Equal Accuracy
- Demographic Parity
- Equal Opportunity

---

## Fairness Metrics Considered

### Equal Accuracy

The model achieved comparable predictive performance across male and female employee groups.

---

### Demographic Parity

Predicted promotion outcomes exhibited similar distributions across protected groups.

---

### Equal Opportunity

Recall values remained comparable between groups, suggesting equal treatment in identifying positive promotion outcomes.

---

## Important Note

This fairness analysis represents an exploratory investigation using one protected attribute (`gender`). Real-world deployments should incorporate:

- Additional protected characteristics
- Calibration metrics
- Equalised odds
- Continuous fairness monitoring
- Human oversight and governance processes

---

# Technologies Used

## Programming Language

- Python

## Data Processing

- Pandas
- NumPy

## Data Visualisation

- Matplotlib
- Seaborn
- Plotly

## Statistical Analysis

- SciPy

## Machine Learning

- Scikit-Learn

## Model Selection

- GridSearchCV
- Cross Validation

---

# Repository Structure

```text
hr-analytics-machine-learning/

├── README.md
├── employee_promotion_fairness_analysis.ipynb
├── requirements.txt
├── data/
│   └── employee_promotion.csv
│
└── images/
    └── fairness_analysis.png
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/hr-analytics-machine-learning.git
cd hr-analytics-machine-learning
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate environment:

### Windows

```bash
venv\Scripts\activate
```

### Mac/Linux

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

---

# Future Improvements

Potential enhancements include:

- Comparing Decision Trees with Random Forests, XGBoost, and LightGBM.
- Incorporating SHAP explainability methods.
- Implementing Equalised Odds and Calibration fairness metrics.
- Building a Streamlit dashboard.
- Deploying the model through FastAPI.
- Creating automated evaluation pipelines.

---

# Key Skills Demonstrated

This project demonstrates proficiency in:

- Exploratory Data Analysis (EDA)
- Data Cleaning
- Missing Value Imputation
- Feature Engineering
- Hyperparameter Optimisation
- Cross Validation
- Model Evaluation
- Responsible AI
- Fairness Analysis
- Human Resource Analytics
- Machine Learning with Scikit-Learn

---

# Author

**Olusegun Samuel Oloro**

MSc Applied Data Science (Distinction) — Teesside University

Aspiring Machine Learning Engineer with interests in:

- Machine Learning
- Responsible AI
- Computer Vision
- Predictive Analytics
- MLOps

LinkedIn: *linkedin.com/in/olusegun-oloro-7243a6b9*

GitHub: *[Add your GitHub URL here](https://github.com/OlusegunOloro)*

---

# License

This project is provided for educational and portfolio purposes.

MIT License.
