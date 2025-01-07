import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
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
# Convert specified columns to categorical
categorical_columns = ['year', 'university', 'school', 'degree']
for col in categorical_columns:
    df[col] = df[col].astype('category')

# Convert other columns to numerical (if possible)
numerical_columns = [col for col in df.columns if col not in categorical_columns]
for col in numerical_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Phase 3: Dataset Exploration (EDA)
print("Dataset Info\n")
df.info()

print("Missing Values:\n", df.isnull().sum())

print("\nSummary Statistics (Numerical Columns):\n", df.describe())

print("\nSummary Statistics (Categorical Columns):")
for col in categorical_columns:
    print(f"\nColumn: {col}")
    print(df[col].value_counts())

# Phase 3.5: Correlation Matrix

# Encode categorical variables
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


numerical_columns = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numerical_columns].corr()
print("\nCorrelation Matrix:\n", correlation_matrix)

# Filter correlations for Basic Monthly Salary - Mean (S$)
target = 'basic_monthly_mean'
correlation_with_target = correlation_matrix[[target]].sort_values(by=target, ascending=False)
print("\nCorrelations with Basic Monthly Salary - Mean (S$):\n", correlation_with_target)

# Plot correlation heatmap for Basic Monthly Salary - Mean (S$)
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_with_target, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
plt.title("Correlation Heatmap with Basic Monthly Salary - Mean (S$)")
plt.show()

# Phase 4: Preprocessing Part 2
df = df.dropna(subset=[target])


# Phase 5: Train/Test Split
X = df.drop(columns=[target])
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Phase 6: Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Phase 7: Model Training and Testing
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Support Vector Regressor': SVR(),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}

# Phase 7.5: Model Evaluation
results = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    results[model_name] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}

# Display evaluation results
results_df = pd.DataFrame(results).T
print("\nModel Evaluation Results:\n", results_df)

# Phase 8: Visualization
plt.figure(figsize=(10, 6))
sns.barplot(x=results_df.index, y='R2', data=results_df.reset_index())
plt.title('Model R2 Scores')
plt.ylabel('R2 Score')
plt.xlabel('Model')
plt.xticks(rotation=45)
plt.show()

# Phase 9: Actual vs Predicted Plot (for the best model)
best_model_name = results_df['R2'].idxmax()
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_best, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title(f'Actual vs Predicted: {best_model_name}')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()
