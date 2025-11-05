import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

# 1. Load Data
print("Loading Bank Churn data...")
# --- IMPORTANT: Assumes bank_churn.csv is in the same 'ml' folder ---
# --- IMPORTANT: Loading CSV from the 'app' folder ---
try:
    # Path is relative from 'ml' folder: Go up (..), then into (app)
    df = pd.read_csv("../app/bank_churn.csv")
except FileNotFoundError:
    print("ERROR: bank_churn.csv not found.")
    print("Please make sure it is inside the 'backend/app' folder.")
    exit()

# Drop columns we don't need for modeling
df = df.drop(columns=['customer_id', 'credit_card'])

# 2. Define Features
X = df.drop('churn', axis=1)  # All columns except 'churn' are features
y = df['churn']              # 'churn' is our target

numeric_features = ['credit_score', 'age', 'tenure', 'balance', 'products_number', 'estimated_salary']
categorical_features = ['country', 'gender', 'active_member']

# 3. Create Preprocessing Pipeline
numeric_transformer = Pipeline(steps=[
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

# 4. Create and Tune the Model (XGBoost)
print("Training and Tuning XGBoost model...")

# Create the pipeline *without* the classifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor)
])

# Define the model we want to tune
xgb = XGBClassifier(use_label_encoder=False, 
                    eval_metric='logloss', 
                    random_state=42)

# --- This is the key ---
# We define a "grid" of parameters to search.
# This will test 3 x 2 x 2 = 12 different model combinations.
# (I'm keeping it small so it runs fast)
param_grid = {
    'classifier__n_estimators': [100, 200, 300],  # How many trees
    'classifier__max_depth': [3, 5],             # How deep the trees are
    'classifier__learning_rate': [0.01, 0.1]     # How fast the model learns
}

# 5. Create the GridSearch object
# This combines the pipeline, the model, and the parameters
# It will use 5-fold cross-validation (cv=5)
# n_jobs=-1 uses all your computer's cores to speed it up
grid_search = GridSearchCV(
    estimator=Pipeline(steps=[('preprocessor', preprocessor), ('classifier', xgb)]),
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=2  # This will print updates so you can see it working
)

# 6. Split data and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Starting Grid Search... This will take a few minutes...")
grid_search.fit(X_train, y_train)

# 7. Evaluate and Save
print(f"Best parameters found: {grid_search.best_params_}")

# Get the best model
best_model = grid_search.best_estimator_

accuracy = best_model.score(X_test, y_test)
print(f"--- NEW TUNED MODEL ACCURACY: {accuracy:.4f} ---")

# Save the *best* pipeline
joblib.dump(best_model, 'bank_churn_pipeline.joblib')
print("New (tuned) model saved as 'bank_churn_pipeline.joblib'")