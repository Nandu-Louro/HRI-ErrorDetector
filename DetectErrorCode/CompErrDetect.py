"""
Experiments starting with the detection of the error, and then its classification
"""

import pandas as pd
import random
import numpy as np
from sklearn.preprocessing import StandardScaler
import statistics
from scipy import stats
from sklearn.metrics import balanced_accuracy_score
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import ndimage

#? General clean of the data
def clean(dataTemp,start, flip):
    #! Remove empty spaces in column names.
    dataTemp.columns = [col.replace(" ", "") for col in dataTemp.columns]
    if start != -1:
        dataTemp = dataTemp[dataTemp.frame>=start]
    dataTemp = dataTemp.drop(columns=['timestamp', 'frame', 'face_id', 'confidence'])
    #! No error type
    dataTemp = dataTemp.drop(columns=['errType'])
    #dataTemp = dataTemp.drop(columns=['TF', 'SNV'])
    #! False = 0; True = 1
    dataTemp = dataTemp.replace(to_replace = True, value = 1)
    dataTemp = dataTemp.replace(to_replace = False, value = 0)
    #! One-hot encode the dataTemp using pandas get_dummies
    dataTemp = pd.get_dummies(dataTemp)
    return dataTemp


def filtering(dataTemp, num):
    #! No emotions
    if(num == 0): 
        dataTemp = dataTemp[dataTemp.columns.drop(list(dataTemp.filter(regex='emotion')))]    
    elif(num == 1):
        dataTemp = dataTemp[dataTemp.columns.drop(list(dataTemp.filter(regex='emotion')))]
        dataTemp = dataTemp[dataTemp.columns.drop(list(dataTemp.filter(regex='lastAction_')))] 
        dataTemp = dataTemp.drop(columns=['t_lastAction'])   
        dataTemp = dataTemp[dataTemp.columns.drop(list(dataTemp.filter(regex='AU')))]
        dataTemp = dataTemp.drop(columns=['speak', 'move'])
    return dataTemp

#? Aquisition of the data
def GetData(randList, samp):
    realsense = pd.DataFrame()
    realsenseTest = pd.DataFrame()

    realsense1 = pd.read_csv('0-13-10-29-31-labeled/0-13-10-29-31-realsense-dataTemp.csv')
    realsense1 = clean(realsense1, 10276, False)
    #realsense1 = clean(realsense1, -1)
    if 1 in randList:
        realsenseTest = pd.concat([realsenseTest,realsense1],ignore_index=True)
    else:
        realsense = pd.concat([realsense,realsense1],ignore_index=True)

    #Hard Rapariga que fica de pé
    realsense2 = pd.read_csv('0-15-14-14-25-labeled/0-15-14-14-25-realsense-dataTemp.csv')
    realsense2 = clean(realsense2, 14206, False)
    #realsense2 = clean(realsense2, -1)
    if 2 in randList:
        realsenseTest = pd.concat([realsenseTest,realsense2],ignore_index=True)
    else:
        realsense = pd.concat([realsense,realsense2],ignore_index=True)
    
    realsense3 = pd.read_csv('0-16-15-58-25-labeled/0-16-15-58-25-realsense-dataTemp.csv')
    realsense3 = clean(realsense3, 14876, False)
    #realsense3 = clean(realsense3, -1)
    if 3 in randList:
        realsenseTest = pd.concat([realsenseTest,realsense3],ignore_index=True)
    else:
        realsense = pd.concat([realsense,realsense3],ignore_index=True)

    realsense4 = pd.read_csv('0-16-16-40-21-labeled/0-16-16-40-21-realsense-dataTemp.csv')
    realsense4 = clean(realsense4, 9765, False)
    #realsense4 = clean(realsense4, -1)
    if 4 in randList:
        realsenseTest = pd.concat([realsenseTest,realsense4],ignore_index=True)
    else:
        realsense = pd.concat([realsense,realsense4],ignore_index=True)

    realsense5 = pd.read_csv('0-17-13-07-53-labeled/0-17-13-07-53-realsense-dataTemp.csv')
    realsense5 = clean(realsense5, 13940, False)
    #realsense5 = clean(realsense5, -1)
    if 5 in randList:
        realsenseTest = pd.concat([realsenseTest,realsense5],ignore_index=True)
    else:
        realsense = pd.concat([realsense,realsense5],ignore_index=True)

    realsense6 = pd.read_csv('0-18-16-40-29-labeled/0-18-16-40-29-realsense-dataTemp.csv')
    realsense6 = clean(realsense6, 15345, False)
    #realsense6 = clean(realsense6, -1)
    if 6 in randList:
        realsenseTest = pd.concat([realsenseTest,realsense6],ignore_index=True)
    else:
        realsense = pd.concat([realsense,realsense6],ignore_index=True)

    realsense7 = pd.read_csv('0-19-10-03-09-labeled/0-19-10-03-09-realsense-dataTemp.csv')
    realsense7 = clean(realsense7, 12762, False)
    #realsense7 = clean(realsense7, -1)
    if 7 in randList:
        realsenseTest = pd.concat([realsenseTest,realsense7],ignore_index=True)
    else:
        realsense = pd.concat([realsense,realsense7],ignore_index=True)

    realsense8 = pd.read_csv('0-19-10-34-41-labeled/0-19-10-34-41-realsense-dataTemp.csv')
    realsense8 = clean(realsense8, 11706, False)
    #realsense8 = clean(realsense8, -1)
    if 8 in randList:
        realsenseTest = pd.concat([realsenseTest,realsense8],ignore_index=True)
    else:
        realsense = pd.concat([realsense,realsense8],ignore_index=True)

    realsense9 = pd.read_csv('0-19-11-27-18-labeled/0-19-11-27-18-realsense-dataTemp.csv')
    realsense9 = clean(realsense9, 9216, False)
    #realsense9 = clean(realsense9, -1)
    if 9 in randList:
        realsenseTest = pd.concat([realsenseTest,realsense9],ignore_index=True)
    else:
        realsense = pd.concat([realsense,realsense9],ignore_index=True)

    realsense10 = pd.read_csv('0-19-12-00-33-labeled/0-19-12-00-33-realsense-dataTemp.csv')
    realsense10 = clean(realsense10, 17574, False)
    #realsense10 = clean(realsense10, -1)
    if 10 in randList:
        realsenseTest = pd.concat([realsenseTest,realsense10],ignore_index=True)
    else:
        realsense = pd.concat([realsense,realsense10],ignore_index=True)

    realsense11 = pd.read_csv('0-19-14-08-50-labeled/0-19-14-08-50-realsense-dataTemp.csv')
    realsense11 = clean(realsense11, 19663, False)
    #realsense11 = clean(realsense11, -1)
    if 11 in randList:
        realsenseTest = pd.concat([realsenseTest,realsense11],ignore_index=True)
    else:
        realsense = pd.concat([realsense,realsense11],ignore_index=True)

    #Hard Long hair, tapa a cara
    realsense12 = pd.read_csv('0-20-11-07-35-labeled/0-20-11-07-35-realsense-dataTemp.csv')
    realsense12 = clean(realsense12, 8936, False)
    #realsense12 = clean(realsense12, -1)
    if 12 in randList:
        realsenseTest = pd.concat([realsenseTest,realsense12],ignore_index=True)
    else:
        realsense = pd.concat([realsense,realsense12],ignore_index=True)
    
    realsense13 = pd.read_csv('0-20-11-35-24-labeled/0-20-11-35-24-realsense-dataTemp.csv')
    realsense13 = clean(realsense13, 1615, False)
    #realsense13 = clean(realsense13, -1)
    if 13 in randList:
        realsenseTest = pd.concat([realsenseTest,realsense13],ignore_index=True)
    else:
        realsense = pd.concat([realsense,realsense13],ignore_index=True)

    realsense14 = pd.read_csv('18-17-37-56-labeled/18-17-37-56-realsense-dataTemp.csv')
    realsense14 = clean(realsense14, 12668, False)
    #realsense18 = clean(realsense18, -1)
    if 14 in randList:
        realsenseTest = pd.concat([realsenseTest,realsense14],ignore_index=True)
    else:
        realsense = pd.concat([realsense,realsense14],ignore_index=True)

    realsense15 = pd.read_csv('12-13-55-45-labeled/12-13-55-45-realsense-dataTemp.csv')
    realsense15 = clean(realsense15, 9064, False)
    if samp == 'o':
        realsense15F = pd.read_csv('12-13-55-45-labeled/12-13-55-45-realsense-dataFlip.csv')
        realsense15F = clean(realsense15F, 9064, True)
    #realsense14 = clean(realsense14, -1)
    if 15 in randList:
        realsenseTest = pd.concat([realsenseTest,realsense15],ignore_index=True)
        if samp == 'o':
            realsenseTest = pd.concat([realsenseTest,realsense15F],ignore_index=True)
    else:
        realsense = pd.concat([realsense,realsense15],ignore_index=True)
        if samp == 'o':
            realsense = pd.concat([realsense,realsense15F],ignore_index=True)

    realsense16 = pd.read_csv('17-11-12-46-labeled/17-11-12-46-realsense-dataTemp.csv')
    realsense16 = clean(realsense16, 12636, False)
    if samp == 'o':
        realsense16F = pd.read_csv('17-11-12-46-labeled/17-11-12-46-realsense-dataFlip.csv')
        realsense16F = clean(realsense16F, 12636, True)
    #realsense15 = clean(realsense15, -1)
    if 16 in randList:
        realsenseTest = pd.concat([realsenseTest,realsense16],ignore_index=True)
        if samp == 'o':
            realsenseTest = pd.concat([realsenseTest,realsense16F],ignore_index=True)
    else:
        realsense = pd.concat([realsense,realsense16],ignore_index=True)
        if samp == 'o':
            realsense = pd.concat([realsense,realsense16F],ignore_index=True)

    realsense17 = pd.read_csv('17-11-42-24-labeled/17-11-42-24-realsense-dataTemp.csv')
    realsense17 = clean(realsense17, 11421, False)
    if samp == 'o':
        realsense17F = pd.read_csv('17-11-42-24-labeled/17-11-42-24-realsense-dataFlip.csv')
        realsense17F = clean(realsense17F, 11421, True)
    #realsense16 = clean(realsense16, -1)
    if 17 in randList:
        realsenseTest = pd.concat([realsenseTest,realsense17],ignore_index=True)
        if samp == 'o':
            realsenseTest = pd.concat([realsenseTest,realsense17F],ignore_index=True)
    else:
        realsense = pd.concat([realsense,realsense17],ignore_index=True)
        if samp == 'o':
            realsense = pd.concat([realsense,realsense17F],ignore_index=True)

    realsense18 = pd.read_csv('18-14-38-57-labeled/18-14-38-57-realsense-dataTemp.csv')
    realsense18 = clean(realsense18, 15259, False)
    if samp == 'o':
        realsense18F = pd.read_csv('18-14-38-57-labeled/18-14-38-57-realsense-dataFlip.csv')
        realsense18F = clean(realsense18F, 15259, True)
    #realsense17 = clean(realsense17, -1)
    if 18 in randList:
        realsenseTest = pd.concat([realsenseTest,realsense18],ignore_index=True)
        if samp == 'o':
            realsenseTest = pd.concat([realsenseTest,realsense18F],ignore_index=True)
    else:
        realsense = pd.concat([realsense,realsense18],ignore_index=True)
        if samp == 'o':
            realsense = pd.concat([realsense,realsense18F],ignore_index=True)



    #Hard fica de lado
    realsense19 = pd.read_csv('19-14-39-37-labeled/19-14-39-37-realsense-dataTemp.csv')
    realsense19 = clean(realsense19, 13350, False)
    if samp == 'o':
        realsense19F = pd.read_csv('19-14-39-37-labeled/19-14-39-37-realsense-dataFlip.csv')
        realsense19F = clean(realsense19F, 13350, True)
    #realsense19 = clean(realsense19, -1)
    if 19 in randList:
        realsenseTest = pd.concat([realsenseTest,realsense19],ignore_index=True)
        if samp == 'o':
            realsenseTest = pd.concat([realsenseTest,realsense19F],ignore_index=True)
    else:
        realsense = pd.concat([realsense,realsense19],ignore_index=True)
        if samp == 'o':
            realsense = pd.concat([realsense,realsense19F],ignore_index=True)
    

    #Hard barba
    realsense20 = pd.read_csv('19-15-28-59-labeled/19-15-28-59-realsense-dataTemp.csv')
    realsense20 = clean(realsense20, 4907, False)
    if samp == 'o':
        realsense20F = pd.read_csv('19-15-28-59-labeled/19-15-28-59-realsense-dataFlip.csv')
        realsense20F = clean(realsense20F, 4907, True)
    #realsense20 = clean(realsense20, -1)
    if 20 in randList:
        realsenseTest = pd.concat([realsenseTest,realsense20],ignore_index=True)
        if samp == 'o':
            realsenseTest = pd.concat([realsenseTest,realsense20F],ignore_index=True)
    else:
        realsense = pd.concat([realsense,realsense20],ignore_index=True)
        if samp == 'o':
            realsense = pd.concat([realsense,realsense20F],ignore_index=True)

    #Hard pastilha
    realsense21 = pd.read_csv('19-16-26-07-labeled/19-16-26-07-realsense-dataTemp.csv')
    realsense21 = clean(realsense21, 4266, False)
    if samp == 'o':
        realsense21F = pd.read_csv('19-16-26-07-labeled/19-16-26-07-realsense-dataFlip.csv')
        realsense21F = clean(realsense21F, 4266, True)
    #realsense21 = clean(realsense21, -1)
    if 21 in randList:
        realsenseTest = pd.concat([realsenseTest,realsense21],ignore_index=True)
        if samp == 'o':
            realsenseTest = pd.concat([realsenseTest,realsense21F],ignore_index=True)
    else:
        realsense = pd.concat([realsense,realsense21],ignore_index=True)
        if samp == 'o':
            realsense = pd.concat([realsense,realsense21F],ignore_index=True)
    
    realsense22 = pd.read_csv('20-10-37-18-labeled/20-10-37-18-realsense-dataTemp.csv')
    realsense22 = clean(realsense22, 10547, False)
    if samp == 'o':
        realsense22F = pd.read_csv('20-10-37-18-labeled/20-10-37-18-realsense-dataFlip.csv')
        realsense22F = clean(realsense22F, 10547, True)
    #realsense22 = clean(realsense22, -1)
    if 22 in randList:
        realsenseTest = pd.concat([realsenseTest,realsense22],ignore_index=True)
        if samp == 'o':
            realsenseTest = pd.concat([realsenseTest,realsense22F],ignore_index=True)
    else:
        realsense = pd.concat([realsense,realsense22],ignore_index=True)
        if samp == 'o':
            realsense = pd.concat([realsense,realsense22F],ignore_index=True)

    realsense23 = pd.read_csv('20-12-03-33-labeled/20-12-03-33-realsense-dataTemp.csv')
    realsense23 = clean(realsense23, 12123, False)
    if samp == 'o':
        realsense23F = pd.read_csv('20-12-03-33-labeled/20-12-03-33-realsense-dataFlip.csv')
        realsense23F = clean(realsense23F, 12123, True)
    #realsense23 = clean(realsense23, -1)
    if 23 in randList:
        realsenseTest = pd.concat([realsenseTest,realsense23],ignore_index=True)
        if samp == 'o':
            realsenseTest = pd.concat([realsenseTest,realsense23F],ignore_index=True)
    else:
        realsense = pd.concat([realsense,realsense23],ignore_index=True)
        if samp == 'o':
            realsense = pd.concat([realsense,realsense23F],ignore_index=True)

    realsense24 = pd.read_csv('20-12-35-06-labeled/20-12-35-06-realsense-dataTemp.csv')
    realsense24 = clean(realsense24, 20283, False)
    if samp == 'o':
        realsense24F = pd.read_csv('20-12-35-06-labeled/20-12-35-06-realsense-dataFlip.csv')
        realsense24F = clean(realsense24F, 20283, True)
    #realsense24 = clean(realsense24, -1)
    if 24 in randList:
        realsenseTest = pd.concat([realsenseTest,realsense24],ignore_index=True)
        if samp == 'o':
            realsenseTest = pd.concat([realsenseTest,realsense24F],ignore_index=True)
    else:
        realsense = pd.concat([realsense,realsense24],ignore_index=True)
        if samp == 'o':
            realsense = pd.concat([realsense,realsense24F],ignore_index=True)
    
    return realsense, realsenseTest


def GetRand():
    randList = []
    x = 0
    while x<3:
        num = random.randint(1, 14)
        if num not in randList:
            randList.append(num)
            x = x + 1
    x = 0
    while x<3:
        num = random.randint(15,24)
        if num not in randList:
            randList.append(num)
            x = x + 1
    return randList


acc = []
err = []
snv = []
tf = []
hl = []
accHG = []
errHG = []
snvHG = []
tfHG = []
hlHG = []
for x in range(30):
    #? Obtain videos for train and test
    num = x + 16
    randList = GetRand()
    #randList = [num]
    realsenseUn, realsenseTestUn = GetData(randList, 'o')

    #? Assure the train and test have the same columns and don't have any NaN

    realsenseUn = realsenseUn.fillna(0)
    realsenseTestUn = realsenseTestUn.fillna(0)
    col_list = list(set().union(realsenseUn.columns, realsenseTestUn.columns))
    realsenseUn = realsenseUn.reindex(columns=col_list, fill_value=0)
    realsenseTestUn = realsenseTestUn.reindex(columns=col_list, fill_value=0)


    #? Split the data for Error detector and type classifier

    realsenseUnType = realsenseUn[realsenseUn.error == 1]
    realsenseUnType = realsenseUnType.drop(columns=['error'])
    realsenseUn = realsenseUn.drop(columns=['TF', 'SNV'])

    #? Balance the training set

    auxNoError = realsenseUn[realsenseUn.error == 0]
    realsense = realsenseUn[realsenseUn.error == 1]
    rows = np.random.choice(auxNoError.index.values, realsense.shape[0])
    aux = auxNoError.loc[rows]
    realsense = pd.concat([realsense,aux],ignore_index=True)
    
    #? Balance the test set

    auxNoError = realsenseTestUn[realsenseTestUn.error == 0]
    realsenseTest = realsenseTestUn[realsenseTestUn.error == 1]
    rows = np.random.choice(auxNoError.index.values, realsenseTest.shape[0])
    aux = auxNoError.loc[rows]
    realsenseTest = pd.concat([realsenseTest,aux],ignore_index=True)
    #realsenseTest = realsenseTestUn

    """ realsensetype = realsenseUnType.loc[(realsenseUnType.TF == 1) & (realsenseUnType.SNV == 1)]
    auxNoError = realsenseUnType.loc[(realsenseUnType.TF == 1) & (realsenseUnType.SNV == 0)]
    realsenseSNV = realsenseUnType.loc[(realsenseUnType.SNV == 1) & (realsenseUnType.TF == 0)]
    rows = np.random.choice(auxNoError.index.values, realsenseSNV.shape[0])
    aux = auxNoError.loc[rows]
    realsensetype = pd.concat([realsensetype,aux],ignore_index=True)
    realsensetype = pd.concat([realsensetype, realsenseSNV], ignore_index=True) """
    realsensetype = realsenseUnType

    #? Randomly shuffle the training sets

    realsense = realsense.sample(frac=1).reset_index(drop=True)
    realsensetype = realsensetype.sample(frac=1).reset_index(drop=True)
    realsenseHG = filtering(realsense, 1)
    realsensetypeHG = filtering(realsensetype, 1)
    realsensetype = filtering(realsensetype, 0)

    #? Organize Train and test

    y_train = np.array(realsense['error'])

    x_train = np.array(realsense.drop(columns=['error']))
    x_trainHG = np.array(realsenseHG.drop(columns=['error']))

    y_trainType = np.array(realsensetype.loc[:,('SNV', 'TF')])

    x_trainType = np.array(realsensetype.drop(columns=['SNV', 'TF']))
    x_trainTypeHG = np.array(realsensetypeHG.drop(columns=['SNV', 'TF']))

    y_test = np.array(realsenseTest.loc[:,('error','SNV', 'TF')])
    aux = realsenseTest.drop(columns=['error','SNV', 'TF'])
    x_test = np.array(aux)
    x_testHG = np.array(filtering(aux,1))
    x_testType = np.array(filtering(aux,0))
    x_testTypeHG = np.array(filtering(aux,1))

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    x_trainType = scaler.fit_transform(x_trainType)
    x_testType = scaler.transform(x_testType)
    x_trainHG = scaler.fit_transform(x_trainHG)
    x_testHG = scaler.transform(x_testHG)
    x_trainTypeHG = scaler.fit_transform(x_trainTypeHG)
    x_testTypeHG = scaler.transform(x_testTypeHG)

    #? Error Detector Train

    rf = RandomForestClassifier(n_estimators = 200, criterion= 'entropy', max_depth=10, max_features='sqrt', min_samples_leaf=15, min_samples_split=15)
    rf.fit(x_train, y_train)
    rfHG = RandomForestClassifier(n_estimators = 200, criterion= 'entropy', max_depth=10, max_features='sqrt', min_samples_leaf=15, min_samples_split=15)
    rfHG.fit(x_trainHG, y_train)

    #? Error Type Classifier Train

    rft = RandomForestClassifier(n_estimators=200, criterion='entropy', max_depth=40, max_features='sqrt', min_samples_leaf=1, min_samples_split=5)
    rft.fit(x_trainType, y_trainType)
    rftHG = RandomForestClassifier(n_estimators=200, criterion='entropy', max_depth=40, max_features='sqrt', min_samples_leaf=1, min_samples_split=5)
    rftHG.fit(x_trainTypeHG, y_trainType)

    #? Test both algorithms in a cascade and sequential way
    prediction = np.empty([x_test.shape[0], 3])
    predictionHG = np.empty([x_testHG.shape[0],3])


    pred = rf.predict(x_test)
    #pred = ndimage.median_filter(pred, size=50)
    print("done prediction of error, let's go with the classification")
    for i in range(pred.shape[0]):
        if int(pred[i]) == 1:
            aux = x_testType[i].reshape(1,-1)
            predt = rft.predict(aux)
            prediction[i, :] = np.c_[1, predt]
        else:
            prediction[i, :] = [0, 0, 0]
    
    predHG = rfHG.predict(x_testHG)
    #predHG = ndimage.median_filter(pred, size=50)
    print("done prediction of error with HG, let's go with the classification")
    for i in range(predHG.shape[0]):
        if int(predHG[i]) == 1:
            aux = x_testTypeHG[i].reshape(1,-1)
            predtHG = rftHG.predict(aux)
            predictionHG[i, :] = np.c_[1, predtHG]
        else:
            predictionHG[i, :] = [0, 0, 0]


    acc.append(accuracy_score(y_test, prediction))
    err.append(balanced_accuracy_score(y_test[:,0], prediction[:,0]))
    #print(err)
    snv.append(balanced_accuracy_score(y_test[:,1], prediction[:,1]))
    #print(snv)
    tf.append(balanced_accuracy_score(y_test[:,2], prediction[:,2]))
    #print(tf)
    hl.append(metrics.hamming_loss(y_test,prediction))
    #print(hl)
    accHG.append(accuracy_score(y_test, predictionHG))
    #print(accHG)
    errHG.append(balanced_accuracy_score(y_test[:,0], predictionHG[:,0]))
    #print(err)
    snvHG.append(balanced_accuracy_score(y_test[:,1], predictionHG[:,1]))
    #print(snv)
    tfHG.append(balanced_accuracy_score(y_test[:,2], predictionHG[:,2]))
    #print(tf)
    hlHG.append(metrics.hamming_loss(y_test,predictionHG))

print("Accuracy vector proposed: ", acc)
print("Hamming loss vector proposed: ", hl)
print("Accuracy vector HG: ", accHG)
print("Hamming loss vector HG: ", hlHG)

print("Accuracy: ", statistics.mean(acc),"(",statistics.stdev(acc),")")
print("Error Detector: ", statistics.mean(err),"(",statistics.stdev(err),")")
print("SNV Classifier: ", statistics.mean(snv),"(",statistics.stdev(snv),")")
print("TF Classifier: ", statistics.mean(tf),"(",statistics.stdev(tf),")")
print("Hamming Loss: ", statistics.mean(hl),"(",statistics.stdev(hl),")")

print("Accuracy HG: ", statistics.mean(accHG),"(",statistics.stdev(accHG),")")
print("Error Detector HG: ", statistics.mean(errHG),"(",statistics.stdev(errHG),")")
print("SNV Classifier HG: ", statistics.mean(snvHG),"(",statistics.stdev(snvHG),")")
print("TF Classifier HG: ", statistics.mean(tfHG),"(",statistics.stdev(tfHG),")")
print("Hamming Loss HG: ", statistics.mean(hlHG),"(",statistics.stdev(hlHG),")")


#! First run
""" Accuracy vector proposed:  [0.6923982344286415, 0.6952975659487702, 0.6857568533969011, 0.7197409256510592, 0.7551724137931034, 0.7259966311061201, 0.7388269611417856, 0.7158284171582842, 0.7042589437819421, 0.6902666321511779, 0.7038977861516722, 0.7383188541911246, 0.673332624188916, 0.7384623398270653, 0.7266469153014988, 0.7469719909159728, 0.757685790365529, 0.7031317964332319, 0.7222001061571125, 0.7299519807923169, 0.7139181286549707, 0.7070855103861349, 0.7274801587301587, 0.7049971607041454, 0.7546836609336609, 0.7186623667975323, 0.7808586762075134, 0.7160573708616413, 0.7383389554495754, 0.7341723874904653]
Hamming loss vector proposed:  [0.1787804479319928, 0.19361964232615436, 0.1993444576877235, 0.16956775963657625, 0.15556278464541315, 0.17699794123151788, 0.16182241205681147, 0.17698230176982302, 0.19627105811092183, 0.1870883308884862, 0.1660975035327367, 0.15815527901698365, 0.1939509981207673, 0.16197867833454874, 0.1680608806785175, 0.1566363865758264, 0.15779068829177761, 0.16661229520081194, 0.17135527246992216, 0.15080032012805122, 0.1532943469785575, 0.18486895204111975, 0.16906415343915343, 0.19670641680863146, 0.14455364455364456, 0.19029257805197233, 0.12780049407956384, 0.17214853157913657, 0.15904772982823465, 0.1744978387998983]
Accuracy vector HG:  [0.5329082883766553, 0.5187332738626227, 0.5587604290822408, 0.5891242747267575, 0.56818477553676, 0.5974733295901179, 0.5783414939056152, 0.5694930506949305, 0.6122089721749007, 0.5560013806195531, 0.5627649552520019, 0.597440713782578, 0.5746197213062441, 0.5725075528700906, 0.5686650400836528, 0.5985364622760535, 0.6096586782861293, 0.557742496737712, 0.5549363057324841, 0.5909963985594238, 0.5554385964912281, 0.5305212182999872, 0.571875, 0.6037478705281091, 0.6035780098280098, 0.5811132922041503, 0.5994122156912854, 0.5795319745497681, 0.5755496103291846, 0.501906941266209]
Hamming loss vector HG:  [0.29911721432074545, 0.3169364088186568, 0.28994835121176005, 0.26476408941663293, 0.2868792019084797, 0.26258656185663487, 0.280306976421155, 0.28177182281771823, 0.26089343176225627, 0.28292633819426466, 0.27622075679070496, 0.2517218439383267, 0.2636421657270503, 0.2791262978782512, 0.28055652375973045, 0.2615022289511313, 0.2652908900185589, 0.2789437436566623, 0.2892560155697098, 0.260984393757503, 0.2815399610136452, 0.3091414978123274, 0.2845238095238095, 0.26647738027635814, 0.2614148239148239, 0.2822723873621238, 0.2580074963795894, 0.2682339408318056, 0.2702880849908883, 0.3341723874904653]
Accuracy:  0.7220132713032674 ( 0.024097376641140207 )
Error Detector:  0.7896686516373036 ( 0.034899380656152675 )
SNV Classifier:  0.8050641386754913 ( 0.04037814660643536 )
TF Classifier:  0.7800575953446168 ( 0.04276153030207362 )
Hamming Loss:  0.17065833749317588 ( 0.01716504516245288 )
Accuracy HG:  0.5723924107548152 ( 0.026882397816632733 )
Error Detector HG:  0.6456680020268153 ( 0.0267993942276858 )
SNV Classifier HG:  0.5177307314115723 ( 0.029750370324682023 )
TF Classifier HG:  0.7058010408387204 ( 0.03296192269847477 )
Hamming Loss HG:  0.27831490087905897 ( 0.018428954319253222 ) """

""" Accuracy vector proposed:  [0.7279925187032419, 0.7289428306141398, 0.698184489453248, 0.7093735280263778, 0.7796447738308203, 0.70587751070135, 0.7081891580161477, 0.7301086394205898, 0.6741304116583342, 0.7118508920562145, 0.7394873271889401, 0.717040312628548, 0.7669097669097669, 0.7299924803953164, 0.731430755510256, 0.7102177554438861, 0.6857078347256718, 0.6836722043144061, 0.6844698685999093, 0.7384977051382514, 0.760989010989011, 0.7211778822117788, 0.7605725439167209, 0.7573948799575541, 0.7343273068795492, 0.7152406417112299, 0.7692675921493538, 0.720315181775628, 0.7407053324835352, 0.7331096894991418]
Hamming loss vector proposed:  [0.1568370739817124, 0.14997568200910819, 0.16615944310350858, 0.16121055110692417, 0.12924865831842575, 0.19108769619141697, 0.19199880387885002, 0.1578892912571133, 0.19451831365457575, 0.1657530150503339, 0.17282706093189965, 0.17753325106266282, 0.13711330377997044, 0.1597736957066638, 0.1450715590096403, 0.1650788665549972, 0.1865733423647174, 0.18336226134391273, 0.18860444041685545, 0.16836449121235866, 0.14034415197205896, 0.17553244675532448, 0.15317718499240945, 0.13295308838484327, 0.15930969711199813, 0.18657835034840384, 0.14285144407212383, 0.16247984604982577, 0.16850789604135685, 0.15192177666822698]
Accuracy vector HG:  [0.5513715710723192, 0.5758058097890967, 0.5994333088466786, 0.5779557230334432, 0.5851648351648352, 0.5366315442871255, 0.5301166218121235, 0.5484738748060011, 0.5771726412083821, 0.5548190969799661, 0.6169834869431644, 0.5736322501028384, 0.5613470613470614, 0.6228918251154797, 0.6291884316473436, 0.5607202680067002, 0.5320049672612328, 0.5693652367964295, 0.4991504304485727, 0.6115526698757416, 0.5566061845131612, 0.5675432456754325, 0.6021470396877033, 0.6113542910200291, 0.6022540502465368, 0.5385877491492465, 0.5775490665390138, 0.4865033546575129, 0.5902910558742299, 0.519893899204244]
Hamming loss vector HG:  [0.2795719035743973, 0.2544988283149843, 0.25000874523384753, 0.2761422515308526, 0.26918817616492036, 0.3120129513774558, 0.30800119612115, 0.2875840662183135, 0.2605396589015353, 0.28771719990697364, 0.2550083205325141, 0.27620663650075417, 0.2723914390581057, 0.23695706663802055, 0.228734021319064, 0.2807265494137353, 0.2909234590200948, 0.26805934374741713, 0.32885893369581637, 0.25291988507033847, 0.28032626288440243, 0.28318834783188346, 0.2574712643678161, 0.23000397930760047, 0.24915864443922672, 0.3077297034516286, 0.2678115525769906, 0.314973734852031, 0.266535656115006, 0.2927653820148749]
Accuracy:  0.7258273608302973 ( 0.026833041155674933 )
Error Detector:  0.8074974955995793 ( 0.032958431302516744 )
SNV Classifier:  0.8032434896324953 ( 0.03922287819999559 )
TF Classifier:  0.8016344461241379 ( 0.04759229875283426 )
Hamming Loss:  0.1640878894444073 ( 0.01833469023208122 )
Accuracy HG:  0.5688837197037215 ( 0.03558972213084952 )
Error Detector HG:  0.6629206024486358 ( 0.041050611667914384 )
SNV Classifier HG:  0.5308954959556611 ( 0.03195960965448799 )
TF Classifier HG:  0.7152967111819013 ( 0.046532567988200824 )
Hamming Loss HG:  0.2742005053393917 ( 0.024907843530006475 ) """

#! Not balanced
""" Accuracy vector proposed:  [0.7290006988120196, 0.6944405933730764, 0.7604369209653437, 0.6721991701244814, 0.7364378558384359, 0.7455265322912382, 0.7892921960072595, 0.7268444798301487, 0.7207052582306132, 0.6777134175291996, 0.7308800709377078, 0.7081855043081602, 0.6959922356702444, 0.7398453727209785, 0.7475397426192278, 0.7322026800670016, 0.7361995753715499, 0.7167985392574558, 0.7268686203490287, 0.7044266294876264, 0.6927392040643523, 0.7180726420620972, 0.6979737479291449, 0.7283313325330132, 0.7551943290149108, 0.7229795520934762, 0.7679245283018868, 0.7289269781638882, 0.7213632404181185, 0.7435219911353563]
Hamming loss vector proposed:  [0.15937572792918706, 0.1645177688432922, 0.16140531472797284, 0.1868440322186966, 0.15352526229097288, 0.16305018510900865, 0.11854204476709014, 0.17137738853503184, 0.15980132105892755, 0.19108663311493052, 0.1635261952264834, 0.17813397533367123, 0.1810344827586207, 0.15974305715824294, 0.1562789132811843, 0.15148659966499162, 0.16527335456475584, 0.16551024548589977, 0.1744594446273735, 0.18429766469153014, 0.20177109793959921, 0.17643038469049013, 0.1898177647508602, 0.1573029211684674, 0.14124500937016216, 0.17481697861444698, 0.14547820429407937, 0.16283165062221178, 0.17006024970963995, 0.14160700079554495]
Accuracy vector HG:  [0.5771488469601677, 0.5180923332871205, 0.6098021274162186, 0.5787771540151331, 0.6219250187990117, 0.5932229535170712, 0.5964609800362977, 0.5796178343949044, 0.5248088625370573, 0.566850251638021, 0.6312347594768344, 0.5228079067410035, 0.6009362868234757, 0.6387606738979922, 0.6037723946505172, 0.6150544388609716, 0.5863190021231423, 0.6394400486914181, 0.6044616397760948, 0.5726734053677239, 0.5811282811176969, 0.5661003710212849, 0.555116605072002, 0.5662064825930372, 0.5667318504033244, 0.6154387103754192, 0.6097592713077423, 0.6060694998826015, 0.583079268292683, 0.5909478349812479]
Hamming loss vector HG:  [0.26319590030281853, 0.31304588936642175, 0.2587212627244653, 0.2932023431779351, 0.23563218390804597, 0.2631804470039764, 0.2617362371445856, 0.2734872611464968, 0.29357154002184427, 0.2734783021555408, 0.23908224340500997, 0.316417469167089, 0.2537489533379006, 0.23294099546118932, 0.26038775338548237, 0.24261934673366833, 0.2669187898089172, 0.22909312233718807, 0.26511908681813195, 0.27884280237016384, 0.2755080440304826, 0.29323699798216496, 0.2917887940189457, 0.28477390956382553, 0.26851625519432903, 0.25594143315662304, 0.25371936673172846, 0.24493230022697035, 0.2686556329849013, 0.24451642232071827]
Accuracy:  0.725618787983568 ( 0.025893942938863535 )
Error Detector:  0.8046112970397454 ( 0.03409070266813617 )
SNV Classifier:  0.7867021499575426 ( 0.04337023828775923 )
TF Classifier:  0.7950930254983574 ( 0.04506672671714408 )
Hamming Loss:  0.16568769577811218 ( 0.01712764411364793 )
Accuracy HG:  0.5874248364685739 ( 0.03131359752001628 )
Error Detector HG:  0.6646260939383878 ( 0.03725071840599418 )
SNV Classifier HG:  0.5212101856855568 ( 0.026635608590220707 )
TF Classifier HG:  0.7198710209143375 ( 0.04180620415984301 )
Hamming Loss HG:  0.266533702866252 ( 0.022332080190202778 ) """

