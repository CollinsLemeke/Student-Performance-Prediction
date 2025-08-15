## Student-Performance-Prediction
This project uses machine learning techniques to predict student performance based . It explores data preprocessing, feature engineering, model selection, and evaluation to build accurate and interpretable predictive models.

This project applies machine learning techniques to predict student performance using a dataset of exam scores and related features. The workflow includes data preprocessing, handling class imbalance with SMOTE, feature scaling, training a Random Forest Classifier, and evaluating the model.

### Model Performance
Accuracy
The model achieved an accuracy of 0.99 — correctly predicting 99% of the instances in the testing set.

### Classification Report
Class	Precision	Recall	F1-score	Support
0	0.98	1.00	0.99	64
1	1.00	0.97	0.99	36

Precision:

Class 0: 0.98

Class 1: 1.00

Recall:

Class 0: 1.00

Class 1: 0.97

F1-score:

Both classes: 0.99

### Confusion Matrix
lua
Copy
Edit
[[64  0]
 [ 1 35]]
True Positives (class 0): 64

True Negatives (class 1): 35

False Positives: 0

False Negatives: 1

### Libraries Used

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

### Dataset
Source: student_exam_data.csv (Kaggle dataset)

The dataset contains exam scores and student attributes used to predict performance classification.

### Workflow
Load dataset using Pandas

Split data into training and testing sets

Handle class imbalance with SMOTE

Scale features using StandardScaler

Train a Random Forest Classifier

Evaluate using accuracy score, classification report, confusion matrix

Visualize results with Seaborn and Matplotlib


### Run the notebook
jupyter notebook notebooks/Student_Performance_Prediction.ipynb

### Conclusion
The model performs extremely well on the dataset, achieving high precision, recall, and F1-scores across both classes with minimal misclassifications.
