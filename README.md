# Bitcoin Price Prediction

## Project Overview

This project aims to predict the price of Bitcoin using the following machine learning models:
- Random Forest
- Support Vector Machine
- XGBoost
- ARIMA
- Prophet
  
The dataset used in this project is the Kaggle dataset [Bitcoin Price Prediction (LightWeight CSV)](https://www.kaggle.com/datasets/team-ai/bitcoin-price-prediction)

## Table of Contents

- [Folder Structure](#folder-structure)
- [Results](#results)
- [Building](#building)
- [License](#license)


## Folder Structure

```plaintext
Bitcoin-Price-Prediction/
│
├── data/
│   ├── case_study_1/
│   │   ├── processed/
│   │   └── raw/
│
├── src/
│   ├── case_study_1/
│   │   ├── model/
│   │   │   ├── arima_functions.py
│   │   │   ├── cross_validation.py
│   │   │   ├── model_evaluation.py
│   │   │   ├── prophet_functions.py
│   │   │   └── walk_forward_validation.py
│   │   ├── preprocessing/
│   │   │   ├── data_cleaning.py
│   │   │   ├── data_preprocessing.py
│   │   │   └── training_models.py
│   │   └── train_evaluate.py
│   │   └── visualization.py
│
├── graphs/
│   ├── case_study_1/
│   │   ├── results/
│   │   │   ├── train_test_split.png
│   │   │   ├── walk_forward_val.png
|
└── config.yaml
└── .gitignore
└── .requirements.txt

```

## Results

The performance of each model is evaluated using multiple metrics and recorded as graphs in graphs/.

### Case Study 1

#### Train, Test Split

| Model         | R²     | MAPE   | MAE      | MSE        |
|---------------|--------|--------|----------|------------|
| Prophet       | 0.368  | 18.449 | 413.669  | 273712.588 |
| XGBoost       | -2.975 | 72.076 | 1278.652 | 1722022.791| 
| Random Forest | -2.990 | 72.251 | 1281.341 | 1728594.680| 
| SVR           | -3.020 | 72.415 | 1285.417 | 1741651.876|
| ARIMA         | -3.761 | 78.558 | 1397.157 | 2062608.181|

### Time Series Cross Validation

| Model         | R²     | MAPE   | MAE      | MSE        |
|---------------|--------|--------|----------|------------|
| Prophet       | 0.812  | 0.068  | 97.419   | 74429.202  |
| SVR           | 0.539  | 0.109  | 164.893  | 211204.798 |
| Random Forest | 0.527  | 0.117  | 168.765  | 206776.694 |
| XGBoost       | 0.469  | 0.131  | 176.450  | 216565.162 |
| ARIMAX        | 0.022  | 0.182  | 202.285  | 236345.004 |

### Walk Forward Validation

| Model         | R²     | MAPE   | MAE      | MSE        |
|---------------|--------|--------|----------|------------|
| ARIMA         | 0.979  | 0.033  | 64.540   | 9058.215   |
| Prophet       | 0.977  | 0.041  | 76.843   | 10018.735  |
| Random Forest | 0.510  | 0.221  | 343.099  | 212202.610 |
| XGBoost       | 0.281  | 0.248  | 374.083  | 311696.006 |
| SVR           | -2.937 | 0.719  | 1273.542 | 1705706.596|


## Building

### Prerequisites
Before building the project, ensure you have Python installed

### Installation

1. **Clone the Repository**:

   Clone the repository to your local machine using the following command:
   
   `git clone https://github.com/RamezzE/Bitcoin-Price-Prediction.git`

2. **Navigate to project folder**

   `cd Bitcoin-Price-Prediction`

3. **Create & activate virtual environment** (Optional but recommended)

    `python -m venv venv`

     `venv\Scripts\activate`
   
4. **Install required packages**

    `pip install -r requirements.txt`

5. **Run main file to train models and view results**

    `python src\case_study_1\train_evaluate.py`

## License 

This project is licensed under the [MIT License](LICENSE).
