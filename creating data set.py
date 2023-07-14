# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
 
 
# Set values for clean data visualization
labelsize   = 12
width       = 5
height      = 4
  
plt.rc('font', family ='serif')
plt.rc('text', usetex = True)
plt.rc('xtick', labelsize = labelsize)
plt.rc('ytick', labelsize = labelsize)
plt.rc('axes', labelsize = labelsize)
  
Class1N = 100
 
# Normal distributed x and y vector with mean 0 and standard deviation 1
x1 = np.random.normal(0, 1, Class1N)
x2 = np.random.normal(0, 1, Class1N)
X = np.stack((x1, x2), axis=0).T
 
 
T1 = np.asarray([[1, 0.5],[0.4,1]])
X_t1 = np.matmul(X,T1)
X_t2 = np.copy(X_t1)
X_t2[:,0]+=3
 
 
X = np.vstack((X_t1, X_t2))
y = np.append(np.ones((Class1N)),np.zeros((Class1N)))
X_t = np.column_stack((X, y))
np.random.shuffle(X_t)
 
 
data = pd.DataFrame(X_t, columns = ['x1','x2','label'])
data.to_csv('./SynData/data.csv')
 
features = data.values[:,:-1]
labels = data.values[:,-1]
 
 
LRC = LogisticRegression()
LRC.fit(features, labels)
 
 
# Retrieve the model parameters.
b = LRC.intercept_[0]
w1, w2 = LRC.coef_.T
 
# Calculate the intercept and gradient of the decision boundary.
c = -b/w2
m = -w1/w2
 
 
u = np.arange(np.min(X_t[:,0]), np.max(X_t[:,0]),0.5)
v = m*u+c
 
fig1, ax = plt.subplots()
plt.plot(u, v, c='k')
scatter = plt.scatter(features[:,0], features[:,1], c=labels)
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="upper left", title="Classes")
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.show()
fig1.set_size_inches(width, height)
fig1.savefig('./Graphs/DataSet.png', dpi=200)
