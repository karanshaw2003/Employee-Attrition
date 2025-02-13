import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
file_path = "WA_Fn-UseC_-HR-Employee-Attrition.csv"
df = pd.read_csv(file_path)

# Exploratory Data Analysis (EDA)
print(df.info())
print(df.describe())
print(df.isnull().sum())

# Visualizations
plt.figure(figsize=(10,5))
sns.countplot(x='Attrition', data=df)
plt.title('Attrition Count')
plt.show()

# Convert target variable 'Attrition' to numeric
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# Drop unnecessary columns
df = df.drop(columns=['EmployeeNumber', 'Over18', 'StandardHours'])

# Identify categorical and numerical columns
categorical_cols = df.select_dtypes(include=['object']).columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop('Attrition')

# Encode categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ], remainder='passthrough'
)

# Split data
X = df.drop(columns=['Attrition'])
y = df['Attrition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train Logistic Regression model
log_reg = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LogisticRegression(max_iter=1000))])
log_reg.fit(X_train, y_train)

# Define and train Random Forest model
rf_clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])
rf_clf.fit(X_train, y_train)

# Save models as pickle files
with open("logistic_regression_model.pkl", "wb") as log_model_file:
    pickle.dump(log_reg, log_model_file)

with open("random_forest_model.pkl", "wb") as rf_model_file:
    pickle.dump(rf_clf, rf_model_file)

# Make predictions
log_reg_pred = log_reg.predict(X_test)
rf_pred = rf_clf.predict(X_test)

# Evaluation function
def evaluate_model(y_true, y_pred, model_name):
    print(f"{model_name} Performance:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"F1-score: {f1_score(y_true, y_pred):.4f}\n")

# Evaluate both models
evaluate_model(y_test, log_reg_pred, "Logistic Regression")
evaluate_model(y_test, rf_pred, "Random Forest")
