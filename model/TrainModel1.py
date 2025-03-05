#SVM model
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
#after preprocess load the data
X_train = pd.read_csv("C:/Users/INFOKOM/Desktop/stage_pfe/baby_cries_classification/preprocessing/X_train.csv").values
X_test = pd.read_csv("C:/Users/INFOKOM/Desktop/stage_pfe/baby_cries_classification/preprocessing/X_test.csv").values
y_train = pd.read_csv("C:/Users/INFOKOM/Desktop/stage_pfe/baby_cries_classification/preprocessing/y_train.csv").values.ravel()
y_test = pd.read_csv("C:/Users/INFOKOM/Desktop/stage_pfe/baby_cries_classification/preprocessing/y_test.csv").values.ravel()
#convert y_train to a pandas series to use method value_counts
yTrainSeries = pd.Series(y_train)
print(yTrainSeries.value_counts())
#Hyperparameter 
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}
gridSearch = GridSearchCV(SVC(probability=True, class_weight='balanced'), param_grid, cv=5)
print("hyperparameter tuning")
gridSearch.fit(X_train, y_train)
#best parameters and best score
print(f"Best parameters: {gridSearch.best_params_}")
print(f"Best cross-validation score: {gridSearch.best_score_}")
#train model with best parameters
svmModel = gridSearch.best_estimator_
print("train SVM")
svmModel.fit(X_train, y_train)
#evaluation
y_pred= svmModel.predict(X_test)
accuracy=accuracy_score(y_test, y_pred)
print(f"SVM accuracy: {accuracy:.4f}")
print("Classification report:")
print(classification_report(y_test, y_pred, zero_division=1))

# Save the Trained Model
joblib.dump(svmModel, "C:/Users/INFOKOM/Desktop/stage_pfe/baby_cries_classification/model/svm_model.pkl")
print("SVM Model saved successfully")