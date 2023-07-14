# Import necessary libraries
import numpy as np
import pandas as pd
from tqdm import tqdm
from numpy import save
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
 
 
# Set values for clean data visualization
labelsize   = 12
width       = 4
height      = 4
  
plt.rc('font', family ='serif')
plt.rc('text', usetex = True)
plt.rc('xtick', labelsize = labelsize)
plt.rc('ytick', labelsize = labelsize)
plt.rc('axes', labelsize = labelsize)
  
 
data = pd.read_csv('./SynData/data.csv').values
 
y = data[:,-1]
X = data[:,:-1]
 
 
numDataTotal = 140
labeledPoolN = 10
batchSz = 10
 
nAccs = (numDataTotal-labeledPoolN)//batchSz
 
 
monteN = 200
  
def computeAccuracy(dataL):
    y_trainC = dataL[:,-1]
    X_trainC = dataL[:,1:-1]
    LRC = LogisticRegression()
    LRC.fit(X_trainC, y_trainC)
    y_pred = LRC.predict(X_test[:,1:])
    Acc = accuracy_score(y_test, y_pred)
    return np.array([[Acc]]), LRC    
 
 
 
def getBatch(dataPool, batchSz):
    dataBatch = dataPool[np.random.choice(dataPool.shape[0], batchSz, replace=False), :]
    remIdx = np.isin(dataPool[:,0], dataBatch[:,0], invert=True)
    dataPool = dataPool[remIdx]
 
    return dataBatch, dataPool
 
 
accuracySmooth = []
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=5)
 
 
for monte in tqdm(range(monteN)):
 
    dataPool = np.hstack((X_train, np.atleast_2d(y_train).T))
 
    dataPoolL = dataPool[np.random.choice(dataPool.shape[0], labeledPoolN, replace=False), :]
 
    remIdx = np.isin(dataPool[:,0], dataPoolL[:,0], invert=True)
 
    dataPool = dataPool[remIdx]
 
    AccuracyRes = np.empty((0,1), float)
 
    accStart, cModel = computeAccuracy(dataPoolL)
     
    AccuracyRes = np.append(AccuracyRes, accStart, axis=0) 
 
    for i in range(nAccs):
        dataBatch, dataPool = getBatch(dataPool, batchSz)
        dataPoolL = np.vstack((dataPoolL, dataBatch))
        cAcc, cModel = computeAccuracy(dataPoolL)
        AccuracyRes = np.append(AccuracyRes, cAcc, axis=0)
    accuracySmooth.append(AccuracyRes) 
 
accuracySmooth = np.asarray(accuracySmooth)
accuracySmooth = np.mean(accuracySmooth, axis=0)
 
 
fig1 = plt.figure()
plt.plot([x for x in range(labeledPoolN, numDataTotal+1, batchSz)], accuracySmooth)
plt.xlabel('Number of samples')
plt.ylabel('accuracy')
plt.show()
 
 
graphData = np.array(([x for x in range(labeledPoolN, numDataTotal+1, batchSz)], accuracySmooth.flatten()))
save('./Graphs/Active_Learning_Batch.npy', graphData)
