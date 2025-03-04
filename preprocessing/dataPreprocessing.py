import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
#loading dataset
fileCSV=r"C:\Users\INFOKOM\Desktop\stage_pfe\baby_cries_classification\features\extracted_features.csv"
df=pd.read_csv(fileCSV)
#Correct Missing Values(exept the label column)
df.iloc[:, :-1] = df.iloc[:, :-1].fillna(df.iloc[:, :-1].mean()) #fillna hadhi bch tchouf missing values w replace them with the value parameter li hwa mean value in this case
#Normalization or Standardize Features
scaler=MinMaxScaler()  #Change to StandardScaler() /MinMaxScaler()
df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])  #Apply to feature columns
# Split Data into Train and Test
X = df.iloc[:, :-1].values  # Features
y = df.iloc[:, -1].values   # Labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#save data after finish processing
pd.DataFrame(X_train).to_csv("C:/Users/INFOKOM/Desktop/stage_pfe/baby_cries_classification/preprocessing/X_train.csv", index=False)
pd.DataFrame(X_test).to_csv("C:/Users/INFOKOM/Desktop/stage_pfe/baby_cries_classification/preprocessing/X_test.csv", index=False)
pd.DataFrame(y_train).to_csv("C:/Users/INFOKOM/Desktop/stage_pfe/baby_cries_classification/preprocessing/y_train.csv", index=False)
pd.DataFrame(y_test).to_csv("C:/Users/INFOKOM/Desktop/stage_pfe/baby_cries_classification/preprocessing/y_test.csv", index=False)

print("Done with Data Preprocessing")