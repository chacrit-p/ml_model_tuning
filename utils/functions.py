from utils.constant import models_save_directory
import os
import joblib


def map_air_quality(value):
    try:
        if value == "Good":
            return 1
        elif value == "Moderate":
            return 2
        elif value == "Poor":
            return 3
        elif value == "Hazardous":
            return 4
    except Exception as e:
        print(f"Error mapping value: {value}. Error: {e}")
        return 0


def save_trained_model(model, model_name, model_type):
    try:
        model_path = os.path.join(models_save_directory, f"{model_name}.{model_type}")
        joblib.dump(model, model_path)
        print("Model saved successfully!")
    except Exception as e:
        print(f"Error saving model {model_name}.{model_type} Error: {e}")
