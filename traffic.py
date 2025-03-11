import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset (make sure to have your dataset in the same directory or provide the correct path)
data = pd.read_csv('traffic.csv')  # Replace 'your_dataset.csv' with your actual dataset filename

# Print the first few rows and the columns
print(data.head())
print(data.columns)

# operate every string in dataset and remove whitespaces
data.columns = data.columns.str.strip()

# Check if 'timestamp' column exists
if 'timestamp' in data.columns:
    # Convert the 'timestamp' column to datetime format with the correct format
    data['timestamp'] = pd.to_datetime(data['timestamp'], format="%d-%m-%Y %H:%M", errors='coerce')
else:
    print("Column 'timestamp' not found in the dataset.")

# Check for any conversion errors
if data['timestamp'].isnull().any():
    print("Some timestamps could not be converted. Check for invalid date formats.")

# checking timestamp data for hour(0-23), week(0-6) and weekend(4-weekdyas/5,6-weekends)
data['hour_of_day'] = data['timestamp'].dt.hour
data['day_of_week'] = data['timestamp'].dt.dayofweek
data['is_weekend'] = data['day_of_week'] >= 5

# specify x and y values to be predicted, y depends on x
X = data[['hour_of_day', 'day_of_week', 'is_weekend']]
y = data['Vehicles']  

# Split the data into training and testing 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest model
model_rf = RandomForestRegressor()

# Hyperparameter tuning using Grid Search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

#subsets divided into 3 parts and evaluated into 3 folds , n_jobs=-1 for parallel processing for hyperparameter tuning
grid_search = GridSearchCV(estimator=model_rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best model from grid search
best_model = grid_search.best_estimator_

# Cross-validation to evaluate the model
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)
print(f'Cross-Validation Scores: {cv_scores}')
print(f'Mean CV Score: {np.mean(cv_scores)}')

# Train the best model
best_model.fit(X_train, y_train)

# Make predictions
predictions_rf = best_model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, predictions_rf)
rmse = np.sqrt(mean_squared_error(y_test, predictions_rf))

print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Squared Error: {rmse}')

# Feature importance(printing the best model)
importances = best_model.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]# A to D and D to A using indexing

# Print feature ranking (number of features (columns) in the dataset)
print("Feature ranking:")
for f in range(X.shape[1]):
    print(f"{f + 1}. {feature_names[indices[f]]} ({importances[indices[f]]})")

# Visualize the predictions
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Actual Traffic Volume', color='blue')
plt.plot(predictions_rf, label='Predicted Traffic Volume', color='orange')
plt.title('Traffic Volume Prediction')
plt.xlabel('Sample Index')
plt.ylabel('Traffic Volume')
plt.legend()
plt.show()