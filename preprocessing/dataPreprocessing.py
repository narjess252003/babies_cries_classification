from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_data(fileCSV):
    # Load dataset
    df = pd.read_csv(fileCSV)
    
    # Correct missing values (except the label column)
    df.iloc[:, :-1] = df.iloc[:, :-1].fillna(df.iloc[:, :-1].mean())  # Replace missing values with mean
    
    # Normalize or standardize features
    scaler = StandardScaler()
    df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])  # Apply to feature columns
    
    # Split data into features and labels
    X = df.iloc[:, :-1].values  # Features
    y = df.iloc[:, -1].values   # Labels
    
    # Apply SMOTE to balance the dataset
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

# Example usage
fileCSV = r"C:\Users\INFOKOM\Desktop\stage_pfe\baby_cries_classification\features\extracted_features.csv"
X_train, X_test, y_train, y_test = preprocess_data(fileCSV)