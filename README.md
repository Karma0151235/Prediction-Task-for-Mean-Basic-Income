### Data-Mining-and-Knowledge-Discovery-Final-Assignment


This project involves comparing the performances of machine learning algorithms on their ability to predict a target variable. These algorithms will be trained on a graduate employment survey dataset to predict the **Basic Monthly Salary - Mean (S$)**. The dataset includes information about graduates from multiple universities, their degrees, and employment statistics. The goal is to predict the average monthly salary based on different features such as university, degree, and other numerical factors.

## Overview

This project focuses on analyzing and predicting the **Basic Monthly Salary - Mean (S$)** for graduates from various universities. The dataset contains both categorical and numerical variables, which are preprocessed and transformed for model training. Multiple machine learning models are trained and evaluated, including Linear Regression, Decision Trees, Random Forest, Support Vector Regressor, and Gradient Boosting. The evaluation metrics are MAE, MSE, RMSE, and R2. 

## How To Run

1. Clone the Repository 
 
 ```
 git clone https://github.com/Karma0151235/Prediction-Task-for-Mean-Basic-Income.git 
 cd Prediction-Task-for-Mean-Basic-Income
 ```

2. Install Required Dependencies 
 
 ```
 pip install -r requirements.txt
 ```

3. Running the Script 
 
 ```
 python main.py
 ```


</br>
Ensure that the CSV is downloaded and in the same folder as the python script (`main.py`) for the script to work. 

## Dataset 
The dataset used in this project is the Graduate Employment Survey dataset retrieved from data.gov.sg, which includes the following columns:

- `year`: The year of graduation. 

- `university`: The university the graduate attended. 

- `school`: The school within the university. 

- `degree`: The degree the graduate obtained. 

- `basic_monthly_mean`: The average monthly salary (target variable). 

- Other numerical and categorical features. 

The dataset is loaded from a CSV file, and missing values are handled appropriately. 

## Technologies Used

**Python**: The primary programming language used for data analysis and machine learning. <br/>
**Pandas**: For data manipulation and preprocessing. <br/>
**NumPy**: For numerical operations. <br/>
**Matplotlib & Seaborn**: For data visualization. <br/>
**Scikit-learn**: For machine learning model training, evaluation, and preprocessing. <br/>

## Steps Involved

### Phase 1: Dataset Loading

 - The dataset is loaded from a CSV file and "Null" values are replaced with `NaN` for proper handling of missing data.


### Phase 2: Preprocessing

 - Convert categorical columns (`year`, `university`, `school`, `degree`) to categorical data types.
 - Convert numerical columns to appropriate numeric types.
 - Encode categorical variables using `LabelEncoder`.


### Phase 3: Exploratory Data Analysis (EDA)

 - Display dataset information, missing values, and summary statistics.
 - Visualize the correlation matrix for numerical features.
 - Display correlations with the target variable (`basic_monthly_mean`).


### Phase 4: Data Cleaning

 - Drop rows with missing target values (`basic_monthly_mean`).
 - Revert categorical variables back to their original form after EDA.


### Phase 5: Model Training

 - Split the data into training and test sets.
 - Scale features using `StandardScaler`.
 - Train multiple models: **Linear Regression**, **Decision Trees**, **Random Forest**, **Support Vector Regressor**, and **Gradient Boosting**.


### Phase 6: Model Evaluation

 - Evaluate each model using metrics like **MAE**, **MSE**, **RMSE**, and **R² score**.
 - Display the evaluation results in a tabular format.
 - Visualize the **R² scores** of the models.


## Results
The models are evaluated based on the following metrics:
</br>

 1. MAE (Mean Absolute Error)
 2. MSE (Mean Squared Error)
 3. RMSE (Root Mean Squared Error)
 4. R² Score: The coefficient of determination.

The best-performing model is identified based on the highest R² score.

## Visualizations
**Correlation Heatmap**: Displays the correlation of features with the target variable (`basic_monthly_mean`). <br/>
**Model R² Scores**: A histogram plot visualizing the **R² scores** of different models. <br/>
**Actual vs Predicted**: A scatter plot comparing the actual and predicted values for the best-performing model. <br/>
