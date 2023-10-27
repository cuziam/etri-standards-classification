# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import joblib

# delete 'space' from data


def normalize_string(s):
    return s.lower().replace(" ", "")


# Load and clean the data
df_cleaned = pd.read_json(
    '/content/drive/MyDrive/etri_drive/data/xyz_selected_data.json').dropna()
df_cleaned['적용단계'] = df_cleaned['적용단계'].str.strip()
label_encoders = {}
for column in df_cleaned.select_dtypes(include=['object']).columns:
    df_cleaned[column] = df_cleaned[column].apply(normalize_string)
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

# Combine the training and test feature sets and labels
all_features = pd.concat([X_train, X_test])
all_labels = pd.concat([y_train, y_test])

# Combine them into a single DataFrame
all_data = pd.concat([all_features, all_labels], axis=1)

# Remove duplicate rows based on the feature columns to get unique combinations
unique_combinations_all = all_data.drop('적용단계', axis=1).drop_duplicates()

# Prepare the corresponding labels for these unique combinations
y_unique_combinations_all = all_data.loc[unique_combinations_all.index, '적용단계']

# Train the Random Forest classifier with best parameters on all unique combinations
best_rf_clf.fit(unique_combinations_all, y_unique_combinations_all)

# Predict probabilities for these unique combinations
predicted_probs_unique_combinations_all = best_rf_clf.predict_proba(
    unique_combinations_all)

# Create a DataFrame to hold the predicted probabilities and convert them to percentage
prob_df_unique_combinations_all = pd.DataFrame(predicted_probs_unique_combinations_all,
                                               columns=label_encoders['적용단계'].inverse_transform(best_rf_clf.classes_)) * 100

# Decode the feature values for easier interpretation
decoded_unique_combinations_all = unique_combinations_all.copy()
for column in decoded_unique_combinations_all.columns:
    decoded_unique_combinations_all[column] = label_encoders[column].inverse_transform(
        decoded_unique_combinations_all[column])

# Combine decoded feature values and predicted probabilities
combined_result_unique_all = pd.concat([decoded_unique_combinations_all.reset_index(
    drop=True), prob_df_unique_combinations_all.reset_index(drop=True)], axis=1)

# Save the DataFrame to an Excel file
excel_file_path_all = '/content/drive/MyDrive/etri_drive/result/predicted_probabilities_all.xlsx'
combined_result_unique_all.to_excel(excel_file_path_all, index=False)
