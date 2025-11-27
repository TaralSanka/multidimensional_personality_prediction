# Multidimensional Personality Cluster Prediction Challenge

A Machine Learning project developed for the **AIT 511 / Machine Learning** course at IIIT Bangalore.
This repository implements preprocessing, feature engineering, and multiple classification models to predict participant personality clusters.

---

## Project Overview

The task is based on the Kaggle competition:
[Multidimensional Personality Cluster Prediction](https://www.kaggle.com/competitions/multidimensional-personality-cluster-prediction)

The dataset contains psychological, behavioral, and demographic features of participants, including:
- **Demographics:** Age, cultural background
- **Traits:** Focus intensity, consistency score, altruism score
- **Behavior:** Hobby engagement, physical activity
- **External:** Upbringing influence, guidance usage

The target variable is **`personality_cluster`**, representing distinct personality profiles.

---

## Libraries Used

| Category | Libraries |
|-----------|------------|
| Data Handling | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Modeling | scikit-learn, tensorflow.keras |
| Utilities | joblib |

---

## Data Preprocessing

Preprocessing steps were designed to handle multi-class classification requirements:

1. **Data Cleaning**
   - Dropped unique identifiers (`participant_id`).
   - Checked for and handled missing values (none found in this dataset).

2. **Feature Encoding**
   - **Categorical:** Label Encoding or One-Hot Encoding for nominal features like `age_group`, `identity_code`, etc.
   - **Target:** Label Encoding for `personality_cluster` (converting classes to integers 0-4).

3. **Feature Scaling**
   - Experimented with **StandardScaler**, **MinMaxScaler**, and **RobustScaler** on numeric features (`focus_intensity`, `consistency_score`).
   - *Key Finding:* Models converged better using **raw, unscaled data**, so scaling was omitted in the final pipeline.

4. **Data Splitting**
   - Split the dataset into 80% training and 20% validation sets.

---

## Models and Tuning

Four classification models were trained and tuned using `GridSearchCV`:

| Model | Best Parameters / Architecture | Kaggle Score |
|--------|----------------|---------------|
| **Logistic Regression** | `C=100, penalty='l2', solver='lbfgs'` | 0.468 |
| **SVM (SVC)** | `C=0.3, kernel='linear', gamma='auto'` | 0.538 |
| **Neural Network (Keras)** | Dense(256) -> Dropout(0.1) -> Dense(128) -> Dense(64) -> Output(5) | 0.589 |
| **MLP Classifier (Sklearn)** | `hidden_layer_sizes=(300, 150, 75)`, `alpha=0.001`, `batch_size=16` | **0.600** |

---

## Best Model

**MLP Classifier (Sklearn)**

**Observation:**
- Linear models (Logistic Regression, SVM with linear kernel) underperformed, indicating complex non-linear relationships between psychological traits and personality clusters.
- Deep learning approaches (Keras NN and Sklearn MLP) captured these complexities effectively.
- The **Sklearn MLP Classifier**, tuned via grid search, achieved the highest generalization performance with a Kaggle score of **0.600**.
