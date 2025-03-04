import librosa
import numpy as np
import os
import pandas as pd 
#first :loading the audio files // we will use librosa
def loading(file):
    y,samplingRate=librosa.load(file,sr=None,mono=True)
    return y,samplingRate 
#extract MFCC features
def extractMfcc(y,sr):
    mfccs=librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13) #The first 13 MFCCs usually contain the most important information for classification
    mfccsMean=np.mean(mfccs,axis=1) #axis 1 means for each row feature mfcc it calculate average and write it in place of all the frames(columns) /to reduce the dimensionality and capture a summary statistic of the audio signal.
    return mfccsMean
#extract chroma features
def extractChroma(y,sr):
    chroma=librosa.feature.chroma_stft(y=y,sr=sr)
    chromaMean=np.mean(chroma,axis=1)
    return chromaMean
#extract spectral contrast features
def extractSpectral(y,sr):
    fmin=sr/80  # Reduce minimum frequency to avoid exceeding Nyquist limit
    spectral_contrast=librosa.feature.spectral_contrast(y=y, sr=sr, fmin=fmin, n_bands=6)
    spectral_contrast_mean = np.mean(spectral_contrast, axis=1)
    return spectral_contrast_mean
#extract zer-crosssing rate feature
def extractZCR(y):
    zcr=librosa.feature.zero_crossing_rate(y=y)
    zcr_mean=np.mean(zcr)  
    return zcr_mean
#process data folder and extract features++Process each .wav file in the folder and extract features and store them
def processDirectory(directory):
    featuresList=[]  #List to store features
    labels=[]  #List to store labels
    #boucle for to go through every folder of data 
    for label in os.listdir(directory):
        class_folder=os.path.join(directory,label)
        #Only process subfolders(classes)
        if os.path.isdir(class_folder):
            #process every .wav file in the folder
            for filename in os.listdir(class_folder):
                if filename.endswith('.wav'):
                    file=os.path.join(class_folder,filename)
                    y,sr=loading(file)
                    mfccs=extractMfcc(y,sr)
                    chroma=extractChroma(y,sr)
                    spectral_contrast=extractSpectral(y,sr)
                    zcr=extractZCR(y)
                    #Combine all features to 1 list
                    features=np.hstack([mfccs,chroma,spectral_contrast,zcr])
                    #Ajout features lel corresponding label
                    featuresList.append(features)
                    labels.append(label)
     #Convert list features and labels to pandas DataFrame
    df=pd.DataFrame(featuresList)
    df['label']=labels  #Ajout labels as last column
    return df
# Function to save the features to a CSV file
def saveCSV(df,csvFile):
    #Save the extracted features to a CSV file.
    #Args:df DataFrame containing features and labels ++ output_path Path where to save the CSV file
    df.to_csv(csvFile,index=False)
def main():
    dataset='./data'  
    csvFile='./features/extracted_features.csv'  
    #Process dataset folder and get features sous format DataFrame
    DFfeatures=processDirectory(dataset)
    #Save the features to a CSV file
    saveCSV(DFfeatures,csvFile)
if __name__=="__main__":
    main()