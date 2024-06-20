# Satchel Hamilton
# Term Final: Kaggle Competetion (Optiver - Trading at the Close)
# DATE: 11/29/2023

# Cell 1
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge

# Load training data
train_data = pd.read_csv('/kaggle/input/optiver-trading-at-the-close/train.csv')
# Remove rows with NaN in the target column
train_data = train_data.dropna(subset=['target'])

# Define features and target
non_feature_columns = ['time_id', 'row_id', 'target']
features = train_data.drop(columns=non_feature_columns)
target = train_data['target']

# Split data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(features, target, test_size=0.2, random_state=42)


# Cell 2
# Preprocessing pipeline
preprocessing_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Models and hyperparameters
models = {
    'RandomForest': RandomForestRegressor(random_state=42),
    'XGBoost': xgb.XGBRegressor(random_state=42),
    'Ridge': Ridge(random_state=42)
}
param_grids = {
    'randomforest': {'randomforestregressor__n_estimators': [10], 'randomforestregressor__max_depth': [3]},
    'xgboost': {'xgbregressor__n_estimators': [10], 'xgbregressor__max_depth': [3]},
    'ridge': {'ridge__alpha': [1]}
}

# Cross-validation and model selection
cv = KFold(n_splits=2, shuffle=True, random_state=42)
best_models = {}

for model_name, model in models.items():
    pipeline = make_pipeline(preprocessing_pipeline, model)
    search = GridSearchCV(pipeline, param_grids[model_name.lower()], cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
    search.fit(X_train, y_train)
    best_models[model_name] = search.best_estimator_

# Model evaluation
for model_name, model in best_models.items():
    predictions = model.predict(X_valid)
    mse = mean_squared_error(y_valid, predictions)
    mae = mean_absolute_error(y_valid, predictions)
    r2 = r2_score(y_valid, predictions)
    print(f"Model: {model_name} - Validation MSE: {mse}, MAE: {mae}, R^2 Score: {r2}")

# Stacking ensemble model
stacking_model = StackingRegressor(estimators=[(name, model) for name, model in best_models.items()], final_estimator=Ridge())
stacking_model.fit(X_train, y_train)

# Model: RandomForest - Validation MSE: 87.87710292511215, MAE: 6.358516506697449, R^2 Score: 0.013408611230841805
# Model: XGBoost - Validation MSE: 87.5795052624826, MAE: 6.344022793850201, R^2 Score: 0.016749723778877845
# Model: Ridge - Validation MSE: 86.9785506933941, MAE: 6.317504146769088, R^2 Score: 0.023496607587842888

# Cell 3
# Import competition module
import optiver2023

# Initialize Kaggle environment and iterate over test set
env = optiver2023.make_env()
api = env.iter_test()

# Define non-feature columns and retrieve feature column names
non_feature_columns = ['time_id', 'row_id', 'target']
feature_columns = train_data.drop(columns=non_feature_columns).columns

for (test_df, revealed_targets, sample_prediction_df) in api:
    # Convert sample_prediction_df to DataFrame if it's a Series
    if isinstance(sample_prediction_df, pd.Series):
        sample_prediction_df = sample_prediction_df.to_frame().T

    # Drop non-feature columns from test data and convert to DataFrame if Series
    test_features = test_df.drop(columns=non_feature_columns, errors='ignore')
    if isinstance(test_features, pd.Series):
        test_features = test_features.to_frame().T

    # Align test data columns with training data and fill missing values
    test_features_aligned = test_features.reindex(columns=feature_columns, fill_value=0)

    # Make predictions using the trained stacking model
    test_predictions = stacking_model.predict(test_features_aligned)

    # Update the sample prediction DataFrame with predictions
    sample_prediction_df['target'] = test_predictions

    # Submit the predictions
    env.predict(sample_prediction_df)

    # Score: 5.4220
    # Rank: 2984/3824
    # Percentile: Percentile = (1 − 2984/3824)×100 = 22%

