import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

# Phase 1: Dataset Loading
df = pd.read_csv("GraduateEmploymentSurveyNTUNUSSITSMUSUSSSUTD.csv")

# Phase 1.5: Handle "Null" as missing values
df.replace("Null", np.nan, inplace=True)

# Phase 2: Preprocessing (before EDA)
categorical_columns = ['year', 'university', 'school', 'degree']
for col in categorical_columns:
    df[col] = df[col].astype('category')

numerical_columns = [col for col in df.columns if col not in categorical_columns]
for col in numerical_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Phase 3.1: Dataset Overview
print("Dataset Info:\n")
df.info()
print("\nMissing Values:\n", df.isnull().sum())

# Phase 3.2: Summary Statistics
print("\nSummary Statistics (Numerical Columns):\n", df.describe())

# Add skewness analysis for numerical columns
numerical_data = df.select_dtypes(include=[np.number])
skewness = numerical_data.skew()
print("\nSkewness Analysis:")
print(skewness)

# Add skewness interpretation
def interpret_skewness(skew_value):
    if abs(skew_value) < 0.5:
        return "approximately symmetric"
    elif abs(skew_value) < 1:
        return "moderately skewed"
    else:
        return "highly skewed"

print("\nSkewness Interpretation:")
for column in numerical_data.columns:
    skew_value = df[column].skew()
    interpretation = interpret_skewness(skew_value)
    print(f"{column}: {skew_value:.2f} ({interpretation})")

print("\nSummary Statistics (Categorical Columns):")
for col in categorical_columns:
    print(f"\nColumn: {col}")
    print(df[col].value_counts())

# Phase 3.3: Correlation Matrix
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

numerical_columns = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numerical_columns].corr()
print("\nCorrelation Matrix:\n", correlation_matrix)

# Filter correlations for target variable
target = 'basic_monthly_mean'
correlation_with_target = correlation_matrix[[target]].sort_values(by=target, ascending=False)
print("\nCorrelations with Basic Monthly Salary - Mean (S$):\n", correlation_with_target)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_with_target, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
plt.title("Correlation Heatmap with Basic Monthly Salary - Mean (S$)")
plt.show()

# Phase 4: Preprocessing Part 2
df = df.dropna(subset=[target])

# Phase 5, 6, 7: Train/Test Splits, Model Training and Evaluation, Visualization
all_results = {}

def evaluate_models(X, y, test_size, partition_label):
    print(f"\nEvaluating Models for Partition: {partition_label} (Train/Test Split: {int((1 - test_size) * 100)}:{int(test_size * 100)})\n")
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Feature Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define parameter grids for each model
    param_grids = {
        'Random Forest': {
            'model': RandomForestRegressor(random_state=42),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'SVR': {
            'model': SVR(),
            'params': {
                'kernel': ['rbf', 'linear'],
                'C': [0.1, 1, 10],
                'epsilon': [0.1, 0.2, 0.3]
            }
        },
        'Gradient Boosting': {
            'model': GradientBoostingRegressor(random_state=42),
            'params': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.3],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10]
            }
        },
        'Decision Tree': {
            'model': DecisionTreeRegressor(random_state=42),
            'params': {
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None] 
            }
        }
    }

    # Linear Regression doesn't require hyperparameter tuning
    models = {'Linear Regression': LinearRegression()}
    
    results = {}
    best_params = {}
    
    # Train and evaluate models
    for name, model_info in param_grids.items():
        print(f"\nTuning {name}...")
        
        # Perform Grid Search
        grid_search = GridSearchCV(
            model_info['model'],
            model_info['params'],
            cv=5,
            scoring='r2',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Store best model and parameters
        models[name] = grid_search.best_estimator_
        best_params[name] = grid_search.best_params_
        
        print(f"Best parameters for {name}:")
        print(grid_search.best_params_)
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

    # Evaluate all models (including Linear Regression)
    for model_name, model in models.items():
        # Train the model if it's Linear Regression (others are already trained)
        if model_name == 'Linear Regression':
            model.fit(X_train, y_train)
        
        # Get predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        results[model_name] = {
            'Train MAE': mean_absolute_error(y_train, y_train_pred),
            'Test MAE': mean_absolute_error(y_test, y_test_pred),
            'Train MSE': mean_squared_error(y_train, y_train_pred),
            'Test MSE': mean_squared_error(y_test, y_test_pred),
            'Train RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'Test RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'Train R2': r2_score(y_train, y_train_pred),
            'Test R2': r2_score(y_test, y_test_pred)
        }

    # Convert results to DataFrame
    results_df = pd.DataFrame(results).T
    print("\nModel Performance Summary:")
    print(results_df)
    
    # Store results and best parameters
    all_results[partition_label] = {
        'metrics': results_df,
        'best_params': best_params
    }

    # Visualization
    results_df[['Train R2', 'Test R2']].plot(kind='bar', figsize=(12, 6), color=['blue', 'orange'])
    plt.title(f'Model R2 Scores (Train vs Test) - {partition_label}')
    plt.ylabel('R2 Score')
    plt.xlabel('Model')
    plt.xticks(rotation=45)
    plt.legend(["Train R2", "Test R2"])
    plt.show()

    # Actual vs Predicted Plot (for the best model)
    best_model_name = results_df['Test R2'].idxmax()
    best_model = models[best_model_name]
    y_pred_best = best_model.predict(X_test)

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_best, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title(f'Actual vs Predicted: {best_model_name} ({partition_label})')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.show()

# Prepare features and target
X = df.drop(columns=[target])
y = df[target]

# Evaluate for different partitions
evaluate_models(X, y, test_size=0.2, partition_label="Partition 1 (80:20)")
evaluate_models(X, y, test_size=0.3, partition_label="Partition 2 (70:30)")
evaluate_models(X, y, test_size=0.4, partition_label="Partition 3 (60:40)")




