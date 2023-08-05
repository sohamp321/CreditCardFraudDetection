# Credit Card Fraud Detection ML Model

This repository contains a machine learning model for credit card fraud detection using Principal Component Analysis (PCA) and three classification algorithms: Random Forest Classifier (RFC), XGBoost, and Logistic Regression (LR). To handle the imbalanced nature of the data, the Synthetic Minority Over-sampling Technique (SMOTE) is applied.

## Overview
Credit card fraud detection is a critical task in the financial industry to identify fraudulent transactions and protect customers from unauthorized activities. This machine learning model aims to identify fraudulent credit card transactions based on historical credit card transaction data.

## Dataset
The dataset used for training and testing the model is the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?select=creditcard.csv) from Kaggle. The dataset contains credit card transaction details, including transaction amount, time, and a class label indicating whether the transaction is fraudulent (Class 1) or not (Class 0).



## Setup
Before running the code, make sure you have the following libraries installed:

- jupyter notebook
- numpy
- pandas
- scikit-learn
- xgboost
- imblearn
- matplotlib
- seaborn


## Code
The code for this credit card fraud detection model is available in the `credit_card_fraud_detection.ipynb` file.

The following steps are performed in the code:

1. Data Loading and Exploration: The dataset is loaded and explored to understand its structure and characteristics.

2. Data Preprocessing: The raw data is preprocessed to handle missing values and normalize the features. Additionally, the dataset is split into training and testing sets.
3. Imbalanced Data Handling: The Synthetic Minority Over-sampling Technique (SMOTE) is applied to balance the classes in the training data, creating synthetic samples for the minority class.
4. Principal Component Analysis (PCA): The high-dimensional feature space is reduced using PCA to capture the most important components while reducing computational complexity.

5. Model Training: Three classification models - Random Forest Classifier, XGBoost, and Logistic Regression - are trained on the preprocessed data.

6. Model Evaluation: The trained models are evaluated on the test dataset using metrics like accuracy, precision, recall, and F1-score to assess their performance.

## How to Use
Clone this repository to your local machine.

Run the credit_card_fraud_detection.ipynb notebook in an IDE of your choice.

The notebook will load the data, preprocess it, perform PCA, train the three models, and display the evaluation results.

## Results
The results of the model evaluation, including accuracy, precision, recall, and F1-score for each model, will be displayed in the notebook itself. Additionally a report file has been attached based on our experiments and their results.

## Acknowledgements
- Yash Shrivastava
- Shreyas Vaidya

## Conclusion
This ML model, using PCA, SMOTE, and three popular classification algorithms, can help financial institutions and businesses detect credit card fraud effectively. The dataset used in this project is from Kaggle, but you can use a similar dataset or replace it with your own data for training and testing. Feel free to experiment with different hyperparameters and data preprocessing techniques to improve the model's performance. Remember to handle imbalanced classes appropriately and take security measures while handling sensitive data. Happy detecting!
