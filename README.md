# E-Commerce Return Prediction & Basket Analysis

This project focuses on predicting customer return behavior based on transaction data and shopping basket compositions using Machine Learning. By analyzing user history and basket values, the goal is to proactively identify high-risk transactions regarding returns to help e-commerce platforms optimize logistics and marketing.

##  Project Structure
* `data/project/train.csv` - Supervised training dataset containing transaction logs.
* `return_prediction.ipynb` - Core Jupyter Notebook containing EDA, preprocessing, model training, and evaluation.
* `README.md` - Project overview.

##  Dataset Features
The model utilizes the following transaction attributes:
* `transactionId`: Unique identifier for each checkout.
* `basket`: Categorized representation of items in the shopping cart.
* `customerType`: Segmentation of users (e.g., `new`, `existing`).
* `totalAmount`: Total monetary value of the transaction.
* `returnLabel` (Target): Binary label (`1` for returned, `0` for kept).

## Workflow & Methodology

### 1. Exploratory Data Analysis (EDA)
* Investigated the distribution of the target variable (`returnLabel`).
* Analyzed the impact of different customer types and basket totals on the likelihood of a return.

### 2. Feature Engineering & Preprocessing
* Handled categorization and structured array-like data from the `basket` column.
* Processed and scaled numerical values like `totalAmount`.
* Encoded categorical attributes (`customerType`).

### 3. Model Training & Comparison
To solve this classification problem, three different algorithms were implemented, fine-tuned using Grid Search, and benchmarked against each other:
1. **Logistic Regression** (Baseline model)
2. **Random Forest Classifier** (Ensemble method)
3. **Gradient Boosting Classifier** (Advanced boosting technique)

### 4. Evaluation & Feature Importance
Models were evaluated using classification metrics (Accuracy, Precision, Recall, F1-Score). Feature importances across all three models were extracted and visually compared using `seaborn` to understand which drivers (like shopping cart content or user status) influence the prediction most.

## Technologies Used
* **Python 3**
* **Data Handling:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (LogisticRegression, RandomForestClassifier, GradientBoostingClassifier, GridSearchCV)
* **Visualization:** Matplotlib, Seaborn
