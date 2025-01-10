import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from utils.constant import features, target, data_set_file_path
from utils.functions import map_air_quality, save_trained_model
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Load dataset
data_set_csv = pd.read_csv(data_set_file_path)

# Preprocess data
data_set_csv["Air Quality"] = data_set_csv["Air Quality"].apply(map_air_quality)
X = data_set_csv[features]
y = data_set_csv[target]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Create KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)  # Default: k=5
knn_model.fit(X_train, y_train)

# Make predictions
y_pred = knn_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(report)

# Save trained model
save_trained_model(knn_model, "knn_model")
