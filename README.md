### Data-Mining-and-Knowledge-Discovery-Final-Assignment


This project involves comparing the performances of machine learning algorithms on their ability to predict a target variable. These algorithms will be trained on a graduate employment survey dataset to predict the **Basic Monthly Salary - Mean (S$)**. The dataset includes information about graduates from multiple universities, their degrees, and employment statistics. The goal is to predict the average monthly salary based on different features such as university, degree, and other numerical factors.

## Overview

This project focuses on analyzing and predicting the **Basic Monthly Salary - Mean (S$)** for graduates from various universities. The dataset contains both categorical and numerical variables, which are preprocessed and transformed for model training. Multiple machine learning models are trained and evaluated, including Linear Regression, Decision Trees, Random Forest, Support Vector Regressor, and Gradient Boosting. The evaluation metrics are MAE, MSE, RMSE, and R2. 

## How To Run
<ol>
 <li> Clone the Repository </li>
 
 ```
 git clone https://github.com/Karma0151235/Prediction-Task-for-Mean-Basic-Income.git 
 cd Prediction-Task-for-Mean-Basic-Income
 ```

 <li> Install Required Dependencies </li>
 
 ```
 pip install -r requirements.txt
 ```

 <li> Running the Script </li>
 
 ```
 python main.py
 ```

</ol>
</br>
Ensure that the CSV is downloaded and in the same folder as the python script ( `main.py` ) for the script to work. 

## Dataset 
The dataset used in this project is the Graduate Employment Survey dataset retrieved from data.gov.sg, which includes the following columns:
<ul>
 <li>`year`: The year of graduation. </li>

 <li>`university`: The university the graduate attended. </li>

 <li>`school`: The school within the university. </li>

 <li>`degree`: The degree the graduate obtained. </li>

 <li>`basic_monthly_mean`: The average monthly salary (target variable). </li>

 <li>Other numerical and categorical features. </li>
</ul>

The dataset is loaded from a CSV file, and missing values are handled appropriately. 

## Technologies Used

**Python**: The primary programming language used for data analysis and machine learning. <br/>
**Pandas**: For data manipulation and preprocessing. <br/>
**NumPy**: For numerical operations. <br/>
**Matplotlib & Seaborn**: For data visualization. <br/>
**Scikit-learn**: For machine learning model training, evaluation, and preprocessing. <br/>

## Steps Involved

### Phase 1: Dataset Loading
<ul>
 <li>The dataset is loaded from a CSV file and "Null" values are replaced with `NaN` for proper handling of missing data.</li>
</ul>

### Phase 2: Preprocessing
<ul>
  <li>Convert categorical columns (`year`, `university`, `school`, `degree`) to categorical data types.</li>
  <li>Convert numerical columns to appropriate numeric types.</li>
  <li>Encode categorical variables using `LabelEncoder`.</li>
</ul>

### Phase 3: Exploratory Data Analysis (EDA)
<ul>
  <li>Display dataset information, missing values, and summary statistics.</li>
  <li>Visualize the correlation matrix for numerical features.</li>
  <li>Display correlations with the target variable (`basic_monthly_mean`).</li>
</ul>

### Phase 4: Data Cleaning
<ul>
  <li>Drop rows with missing target values (`basic_monthly_mean`).</li>
  <li>Revert categorical variables back to their original form after EDA.</li>
</ul>

### Phase 5: Model Training
<ul>
  <li>Split the data into training and test sets.</li>
  <li>Scale features using `StandardScaler`.</li>
  <li>Train multiple models: <strong>Linear Regression</strong>, <strong>Decision Trees</strong>, <strong>Random Forest</strong>, <strong>Support Vector Regressor</strong>, and <strong>Gradient Boosting</strong>.</li>
</ul>

### Phase 6: Model Evaluation
<ul>
 <li>Evaluate each model using metrics like <strong>MAE</strong>, <strong>MSE</strong>, <strong>RMSE</strong>, and <strong>R² score</strong>.</li>
 <li>Display the evaluation results in a tabular format.</li>
 <li>Visualize the <strong>R² scores</strong> of the models.</li>
</ul>

## Results
The models are evaluated based on the following metrics:
</br>
<ol>
  <li>MAE (Mean Absolute Error)</li>
  <li>MSE (Mean Squared Error)</li>
  <li>RMSE (Root Mean Squared Error)</li>
  <li>R² Score: The coefficient of determination.</li>
</ol>
The best-performing model is identified based on the highest R² score.

## Visualizations
**Correlation Heatmap**: Displays the correlation of features with the target variable (`basic_monthly_mean`). <br/>
**Model R² Scores**: A histogram plot visualizing the **R² scores** of different models. <br/>
**Actual vs Predicted**: A scatter plot comparing the actual and predicted values for the best-performing model. <br/>
