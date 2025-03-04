# SVM model
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

# Load data after Preprocessing
X_train = pd.read_csv("C:/Users/INFOKOM/Desktop/stage_pfe/baby_cries_classification/preprocessing/X_train.csv").values
X_test = pd.read_csv("C:/Users/INFOKOM/Desktop/stage_pfe/baby_cries_classification/preprocessing/X_test.csv").values
y_train = pd.read_csv("C:/Users/INFOKOM/Desktop/stage_pfe/baby_cries_classification/preprocessing/y_train.csv").values.ravel()
y_test = pd.read_csv("C:/Users/INFOKOM/Desktop/stage_pfe/baby_cries_classification/preprocessing/y_test.csv").values.ravel()

# Convert y_train to a Pandas Series to use value_counts
y_train_series = pd.Series(y_train)
print(y_train_series.value_counts())

# Hyperparameter tuning using Grid Search
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

grid_search = GridSearchCV(SVC(probability=True, class_weight='balanced'), param_grid, cv=5)
print("Start hyperparameter tuning...")
grid_search.fit(X_train, y_train)

# Print the best parameters and best score
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_}")

# Train the model with the best parameters
svm_model = grid_search.best_estimator_
print("Start training SVM with best parameters...")
svm_model.fit(X_train, y_train)

# Evaluation of SVM
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"SVM Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=1))

# Save the Trained Model
joblib.dump(svm_model, "C:/Users/INFOKOM/Desktop/stage_pfe/baby_cries_classification/model/svm_model.pkl")
print("SVM Model saved successfully")