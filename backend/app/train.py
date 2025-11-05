import pandas as pd
import joblib
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer  # <-- NEW IMPORT
from sklearn.linear_model import LogisticRegression


# 1. Load Data
print("Loading data...")
X, y = fetch_openml(data_id=42178, as_frame=True, return_X_y=True, parser='pandas')

# We ONLY drop customerID now
X = X.drop(columns=['customerID'], errors='ignore')
y = y.map({'No': 0, 'Yes': 1})

# --- KEY UPGRADE ---
# TotalCharges is a string, needs to be numeric
# We force errors to NaN, which our imputer will fix
X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')

# 2. Define Features
numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges'] # <-- ADDED TotalCharges
categorical_features = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod'
]
all_features = numeric_features + categorical_features
X = X[all_features]

# 3. Create Preprocessing Pipeline
# --- KEY UPGRADE: Numeric transformer now imputes! ---
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), # <-- FILLS MISSING VALUES
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 4. Create and Train the Model (XGBoost)
print("Training LogisticRegression model with Imputed TotalCharges...")
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

# Split data and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_pipeline.fit(X_train, y_train)

# 5. Evaluate and Save
accuracy = model_pipeline.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.4f}") # <-- THIS SHOULD BE MUCH BETTER

# Save the new, more powerful pipeline
joblib.dump(model_pipeline, 'churn_pipeline.joblib')
print("New model saved as 'churn_pipeline.joblib'")