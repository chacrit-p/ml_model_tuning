import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from utils.constant import features, target, data_set_file_path
from utils.functions import map_air_quality, save_trained_model
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

data_set_csv = pd.read_csv(data_set_file_path)

data_set_csv["Air Quality"] = data_set_csv["Air Quality"].apply(map_air_quality)
X = data_set_csv[features]
y = data_set_csv[target]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

logistic_regression_model = LogisticRegression(max_iter=5001, solver="saga")
logistic_regression_model.fit(X, y)

y_pred = logistic_regression_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(report)

save_trained_model(logistic_regression_model, "logistic_regression_model")
