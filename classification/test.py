import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import time
from utils.constant import features, target, data_set_file_path
from utils.functions import map_air_quality

# Load dataset
data = pd.read_csv(data_set_file_path)
data[target] = data[target].apply(map_air_quality)

# Features and target
X = data[features]
y = data[target]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Initialize models
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": GaussianNB(),
}


# ANN Model
def build_ann():
    model = Sequential(
        [
            Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
            Dense(32, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# Evaluate models
results = []
for name, model in models.items():
    start_time = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results.append(
        {"Model": name, "Accuracy": accuracy, "Time": time.time() - start_time}
    )

# ANN
start_time = time.time()
ann_model = build_ann()
ann_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
y_pred_ann = (ann_model.predict(X_test) > 0.5).astype(int)
accuracy_ann = accuracy_score(y_test, y_pred_ann)
results.append(
    {"Model": "ANN", "Accuracy": accuracy_ann, "Time": time.time() - start_time}
)

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Display results
print(results_df)

# Plot results
plt.bar(results_df["Model"], results_df["Accuracy"], color="skyblue")
plt.title("Model Comparison: Accuracy")
plt.ylabel("Accuracy")
plt.show()
