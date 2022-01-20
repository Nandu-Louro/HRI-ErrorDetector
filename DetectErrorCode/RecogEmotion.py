import pandas as pd 
import numpy as np


# Classifies each frame with emotion acording to the Action Units. based on the correspondance of Ekman et al. 1976
def getdataEmo(data):
    Surprise = data.filter(['AU01_r', 'AU02_r', 'AU05_r', 'AU26_r'],  axis=1) 
    Surprise['surprise'] = Surprise.sum(axis=1)/len(Surprise.columns)

    Fear = data.filter(['AU01_r', 'AU02_r','AU04_r','AU05_r', 'AU20_r', 'AU26_r'],axis=1) 
    Fear['fear'] = Fear.sum(axis=1)/len(Fear.columns)

    Disgust = data.filter(['AU09_r', 'AU15_r', 'AU16_r'],axis=1) 
    Disgust['disgust'] = Disgust.sum(axis=1)/len(Disgust.columns)

    Anger = data.filter(['AU04_r', 'AU07_r', 'AU05_r', 'AU23_r'],axis=1) 
    Anger['angry'] = Anger.sum(axis=1)/len(Anger.columns)

    Happiness = data.filter(['AU06_r','AU12_r'],axis=1) 
    Happiness['happy'] = Happiness.sum(axis=1)/len(Happiness.columns)

    Sadness = data.filter(['AU01_r','AU04_r','AU15_r'],axis=1)
    Sadness['sad']= Sadness.sum(axis=1)/len(Sadness.columns)

    Mine = Happiness.filter(['happy'],axis=1)
    Mine = Mine.join(Sadness.loc[:,['sad']])
    Mine = Mine.join(Anger.loc[:,['angry']])
    Mine = Mine.join(Disgust.loc[:,['disgust']])
    Mine = Mine.join(Fear.loc[:,['fear']])
    Mine = Mine.join(Surprise.loc[:,['surprise']])
    Mine['emotion'] = Mine.idxmax(axis=1)
    Mine.loc[Mine.max(axis=1)<0.8, 'emotion'] = 'neutral'

    return Mine


# Load csv file from openFace containing AU information
dataR = pd.read_csv('20-12-35-06-labeled/20-12-35-06-realsense-datatemp.csv')
dataL = pd.read_csv('20-12-35-06-labeled/20-12-35-06-laptop-datatemp.csv')

# Sends file to emotion function and returns a dataframe with each frame classified with emotion
MineR = getdataEmo(dataR)
MineL = getdataEmo(dataL)
dataR['emotion'] = MineR.loc[:,['emotion']]
dataL['emotion'] = MineL.loc[:,['emotion']]


