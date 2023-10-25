
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import joblib


# Load and clean the data
df_cleaned = pd.read_json('data/xyz_selected_data.json').dropna()
label_encoders = {}
for column in df_cleaned.columns:
    le = LabelEncoder()
    df_cleaned[column] = le.fit_transform(df_cleaned[column])
    label_encoders[column] = le
# Split data into training and test sets
X = df_cleaned.drop('적용단계', axis=1)
y = df_cleaned['적용단계']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Initialize Random Forest classifier
rf_clf = RandomForestClassifier(random_state=42)

# Hyperparameter Tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
grid_search = GridSearchCV(
    estimator=rf_clf, param_grid=param_grid, cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_

# Train and evaluate Random Forest with best parameters
best_rf_clf = RandomForestClassifier(**best_params, random_state=42)
best_rf_clf.fit(X_train, y_train)
y_pred = best_rf_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Perform 5-Fold Cross-Validation
cv_scores = cross_val_score(best_rf_clf, X, y, cv=5)
cv_mean = cv_scores.mean()
cv_std = cv_scores.std()


# Save the trained model
model_file_path = 'models/model.joblib'
joblib.dump(best_rf_clf, model_file_path)


# Load the trained model from the file
loaded_model = joblib.load(model_file_path)

# Select a few samples from the test set
sample_data = X_test.sample(5, random_state=42)

# Make predictions using the loaded model
sample_predictions = loaded_model.predict(sample_data)

# Decode the predicted labels back to original labels
decoded_predictions = label_encoders['적용단계'].inverse_transform(
    sample_predictions)

# Show the sample data and decoded predictions
sample_data, decoded_predictions
