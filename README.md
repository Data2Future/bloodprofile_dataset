# Blood Profile Dataset for Data Science & Machine Learning

![Blood Profile](https://via.placeholder.com/1200x400.png?text=Blood+Profile+Data+Visualization)

## Overview
This project provides a **synthetic blood profile dataset** for **Data Scientists** and **ML learners** to explore, analyze, and build predictive models. The dataset simulates blood test values for 1,000 patients, which can be used for tasks like **classification**, **regression**, **anomaly detection**, and **clustering**.

### Dataset Features:
- **Patient_ID**: Unique identifier for each patient.
- **Age**: Patient's age (18-85 years).
- **Gender**: Male/Female.
- **Hemoglobin (g/dL)**: Hemoglobin levels.
- **WBC (10^3/uL)**: White blood cell count.
- **RBC (million/uL)**: Red blood cell count.
- **Platelets (10^3/uL)**: Platelet count.
- **MCV (fL)**: Mean Corpuscular Volume.
- **MCH (pg)**: Mean Corpuscular Hemoglobin.
- **MCHC (g/dL)**: Mean Corpuscular Hemoglobin Concentration.
- **RDW (%)**: Red Cell Distribution Width.

---

## What Can You Do With This Dataset?

This dataset offers a variety of data science and machine learning tasks such as:

### 1. Data Exploration & Visualization

#### Techniques:
- **Descriptive statistics**: To understand feature distributions.
- **Correlation analysis**: To study relationships between variables.
- **Data visualization**: Use histograms, box plots, pair plots, and heatmaps.

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('synthetic_blood_profile_dataset.csv')

# Descriptive statistics
print(df.describe())

# Plot correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Blood Parameters')
plt.show()
```

---

### 2. Machine Learning Tasks

Below are some **ML code samples** to train models and predict based on blood profiles.

### a. Disease Prediction (Classification)

You can build a **classification model** to predict if a patient has a specific condition like **anemia** based on blood test results.

```python
# Disease Prediction Example - Logistic Regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Sample: Create a binary target for "Anemia"
df['Anemia'] = df['Hemoglobin (g/dL)'].apply(lambda x: 1 if x < 12 else 0)

# Define features and target
X = df[['Age', 'WBC (10^3/uL)', 'RBC (million/uL)', 'Platelets (10^3/uL)', 'MCV (fL)', 'MCH (pg)', 'MCHC (g/dL)', 'RDW (%)']]
y = df['Anemia']

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

### Sample Visualizations:
- **Confusion Matrix**:
  ![Confusion Matrix](https://via.placeholder.com/500x300.png?text=Confusion+Matrix+for+Anemia+Prediction)
  
---

### b. Regression Analysis (Predict Continuous Variables)

Use regression techniques to predict blood parameters like **Hemoglobin** levels based on other blood tests.

```python
# Predict Hemoglobin levels using Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Define features and target (Hemoglobin)
X = df[['Age', 'WBC (10^3/uL)', 'RBC (million/uL)', 'Platelets (10^3/uL)', 'MCV (fL)', 'MCH (pg)', 'MCHC (g/dL)', 'RDW (%)']]
y = df['Hemoglobin (g/dL)']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest Regressor
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_train)

# Make predictions
y_pred = regressor.predict(X_test)

# Evaluate the model
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
```

### Sample Visualizations:
- **Predicted vs. Actual Hemoglobin Levels**:
  ![Regression Plot](https://via.placeholder.com/500x300.png?text=Predicted+vs+Actual+Hemoglobin+Levels)

---

### c. Clustering (Patient Segmentation)

Use **K-Means Clustering** to group patients based on their blood profile similarity.

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Define features for clustering
X = df[['Age', 'WBC (10^3/uL)', 'RBC (million/uL)', 'Platelets (10^3/uL)', 'MCV (fL)', 'MCH (pg)', 'MCHC (g/dL)', 'RDW (%)']]

# Train K-Means with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# Visualize clusters based on two features
plt.figure(figsize=(8,6))
plt.scatter(df['WBC (10^3/uL)'], df['RBC (million/uL)'], c=df['Cluster'], cmap='viridis')
plt.title('Patient Segmentation Based on Blood Profiles')
plt.xlabel('WBC (10^3/uL)')
plt.ylabel('RBC (million/uL)')
plt.show()
```

### Sample Visualizations:
- **Cluster Analysis**:
  ![K-Means Clustering](https://via.placeholder.com/500x300.png?text=K-Means+Clustering+Visualization)

---

### d. Anomaly Detection (Outlier Detection)

Detect anomalies (e.g., abnormal blood test results) using **Isolation Forest** or **One-Class SVM**.

```python
from sklearn.ensemble import IsolationForest

# Define features for anomaly detection
X = df[['Age', 'WBC (10^3/uL)', 'RBC (million/uL)', 'Platelets (10^3/uL)', 'MCV (fL)', 'MCH (pg)', 'MCHC (g/dL)', 'RDW (%)']]

# Train an Isolation Forest
isolation_forest = IsolationForest(contamination=0.05, random_state=42)
df['Anomaly'] = isolation_forest.fit_predict(X)

# Visualize anomalies
plt.figure(figsize=(8,6))
plt.scatter(df['WBC (10^3/uL)'], df['RBC (million/uL)'], c=df['Anomaly'], cmap='coolwarm')
plt.title('Anomaly Detection in Blood Profiles')
plt.xlabel('WBC (10^3/uL)')
plt.ylabel('RBC (million/uL)')
plt.show()
```

### Sample Visualizations:
- **Anomaly Detection**:
  ![Anomaly Detection](https://via.placeholder.com/500x300.png?text=Anomaly+Detection+in+Blood+Profile)

---

## How to Use This Dataset

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-repo/blood-profile-dataset.git
cd blood-profile-dataset
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Generate the Dataset
```bash
python generate_blood_profile.py
```

The dataset will be saved in **CSV**, **Parquet**, or **Avro** formats.

---

## Contribution Guidelines

Contributions are welcome! Feel free to submit pull requests with new features, improvements, or additional datasets.



With these **machine learning examples** and **code snippets**, users can quickly get started with their own analysis or models on the **Blood Profile Dataset**!

