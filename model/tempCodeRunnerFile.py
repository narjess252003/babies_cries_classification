# Random Forest model
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load data after Preprocessing
X_train = pd.read_csv("C:/Users/INFOKOM/Desktop/stage_pfe/baby_cries_classification/preprocessing/X_train.csv").values
X_test = pd.read_csv("C:/Users/INFOKOM/Desktop/stage_pfe/baby_cries_classification/preprocessing/X_test.csv").values
y_train = pd.read_csv("C:/Users/INFOKOM/Desktop/stage_pfe/baby_cries_classification/preprocessing/y_train.csv").values.ravel()
y_test = pd.read_csv("C:/Users/INFOKOM/Desktop/stage_pfe/baby_cries_classification/preprocessing/y_test.csv").values.ravel()

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
print("Train Random Forest")
rf_model.fit(X_train, y_train)

# Evaluate the Model
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=1))

# Save the Trained Model
output_dir = "C:/Users/INFOKOM/Desktop/stage_pfe/baby_cries_classification/model"
os.makedirs(output_dir, exist_ok=True)
joblib.dump(rf_model, f"{output_dir}/random_forest_model.pkl")
print("Random Forest saved ")