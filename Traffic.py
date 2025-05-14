import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("traffic.csv")
print(df.head(5))

# Convert timestamp to datetime format
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d-%m-%Y %H:%M')

# Extract features from timestamp
df['hour'] = df['timestamp'].dt.hour
df['day'] = df['timestamp'].dt.day
df['month'] = df['timestamp'].dt.month
df['week_day'] = df['timestamp'].dt.weekday

# Define features and target
X = df[['Junction', 'hour', 'day', 'month', 'week_day']]
y = df['Vehicles']  # Traffic flow prediction as regression

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Bar Graph of Mean Traffic Flow per Junction
plt.figure(figsize=(8, 6))
df.groupby('Junction')['Vehicles'].mean().plot(kind='bar', color='blue')
plt.xlabel('Junction')
plt.ylabel('Average Vehicles')
plt.title('Average Traffic Flow per Junction')
plt.show()

# accuracy 
accuracy = r2_score(y_test, y_pred) * 100
print(f"Model Accuracy: {accuracy:.2f}%")

# Accuracy Graph: 
plt.figure(figsize=(6, 4))
plt.bar(["Model Accuracy"], [r2_score(y_test, y_pred) * 100], color='blue')
plt.ylim(0, 100)  # Set y-axis limit to 100% for clarity
plt.ylabel("Accuracy (%)")
plt.title("Random Forest Regressor Accuracy")
plt.show()

bar_width = 0.4
plt.figure(figsize=(10, 6))
x = np.arange(len(y_test))
plt.bar(x - bar_width/2, y_test, width=bar_width, label='Actual', alpha=0.7)
plt.bar(x + bar_width/2, y_pred, width=bar_width, label='Predicted', alpha=0.7)
plt.xlabel('Sample Index')
plt.ylabel('Number of Vehicles')
plt.title('Actual vs Predicted Traffic Flow')
plt.legend()
plt.tight_layout()
plt.show()
