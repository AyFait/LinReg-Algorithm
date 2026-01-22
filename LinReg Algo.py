import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mpl
import matplotlib.pyplot as plt
import re
np.set_printoptions(precision=2)  # reduced display precision on numpy arrays


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



inputFilepath = '/kaggle/input/house-prices-art-cleaned/HP-AdvRegTech-TrainCleaned.csv'
#testFilepath = '/kaggle/input/house-prices-art-cleaned/HPTestCleanedScaledMN.csv'
outputFilepath = '/kaggle/working/HPriceARTSubmission.csv'
train = pd.read_csv(inputFilepath)#or np.getfrmtxt(trainFile, delimiter=',', skip_headers=1)
#test = pd.read_csv(testFilepath)
print(train.dtypes)#Data type of each col
yTrainCol = train.columns[-1]#Change value for target col
yTrain = train[yTrainCol]
#print(yTrain)
train.drop(yTrainCol, axis=1, inplace=True)
xTrain = train.to_numpy()
yTrain = yTrain.to_numpy()
#print(yTrain)
m, n = xTrain.shape
print(f'Num of training examples, m: {len(xTrain)}')
print(f'Num of training features, n: {len(xTrain[0])}')



#Model f(x) = w.x+b: returns f(x) as an array
def linRegModel(xTrain, w, b):#for each training example
    m, n = xTrain.shape#or len(xTrain) or m = xTrain.shape[0]
    f_x = []
    for i in range(m):#Each training examples with n features
    
        #Using iteration over each elmt in a row
        #sum_wx = 0
        #for j in range(n):#Each training features
        #    w_x = w[j] * xTrain[i, j]#running through each feature and its w 
        #    sum_wx += w_x
        #f_xi = sum_wx + b

        #Using vectorization 
        f_xi = np.dot(w, xTrain[i]) + b
        f_x.append(f_xi)
    return f_x




#Cost Fxn J(w,b) = 1/2m * sum((f(x) - y)**2): returns a single val j_wb
def costFunction(f_x, yTrain):#or (xTrain, yTrain, w, b)
    m, n = xTrain.shape#or len(xTrain) or m = xTrain.shape[0]
    sumSqrdErr = 0
    for i in range(m):
        sqrdErr_i = (f_x[i] - yTrain[i])**2
        #sqrdErr_i = ((np.dot(w, xTrain[i]) + b) - yTrain[i])**2 
        sumSqrdErr += sqrdErr_i
    j_wb = sumSqrdErr / (2 * m) #scalar
    return j_wb



#Derivative j_dw, j_db
def derivatives(f_x, xTrain, yTrain):#or (xTrain, yTrain, w, b)
    m, n = xTrain.shape#or len(xTrain) or m = xTrain.shape[0]
    j_dw = []
    sumErrB = 0
    sumErrW = np.zeros((n,))
    for i in range(m):#for each w in each training example
        Err_i = (f_x[i] - yTrain[i])
        #Err_i = ((np.dot(w, xTrain[i]) + b) - yTrain[i]) 
        sumErrB += Err_i #b for each t-exp
        for j in range(n):#for each w in each training feat 
            sumErrW[j] = sumErrW[j] + Err_i * xTrain[i, j]#OR
            #j_dw.append(Err_i * xTrain[i, j])#doest give correct output dont know why

    #j_dw = np.array(j_dw) / m
    sumErrW = sumErrW / m
    sumErrB = sumErrB / m
    j_dw = sumErrW #1D array for each training example  
    j_db = sumErrB #scalar                             
    
    return j_dw, j_db



#GradDesc
def gardientDescent(w1, b1, alpha, j_dw, j_db, iters):
    m, n = xTrain.shape#or len(xTrain) or m = xTrain.shape[0]
    w = np.zeros_like(w1)
    #w = copy.deepcopy(w1)  #avoid modifying global w within function
    b = 0
    for itr in range(iters):
        #for j in range(n):
        #    w[j] = w1[j] - (alpha * j_dw[j])
        w = w1 - (alpha * j_dw)
        b = b1 - (alpha * j_db)
    
    return w, b



# Plot the data points
def plotData(x_Train, y_Train):
    plt.scatter(x_Train, y_Train, marker='x', c='r')
    # Set the title
    plt.title("Housing Prices")
    # Set the y-axis label
    plt.ylabel('Price (in 1000s of dollars)')
    # Set the x-axis label
    plt.xlabel('Size (1000 sqft)')
    pData = plt.show()

    #pass
    return pData



# Plot the prediction
def plotPred(x_Train, y_Train, w1):
    # Plot our model prediction
    tmp_f_wb = w1
    plt.plot(x_Train, tmp_f_wb, c='b',label='Our Prediction')
    
    # Plot the data points
    plt.scatter(x_Train, y_Train, marker='x', c='r',label='Actual Values')
    
    # Set the title
    plt.title("Housing Prices")
    # Set the y-axis label
    plt.ylabel('Price (in 1000s of dollars)')
    # Set the x-axis label
    plt.xlabel('Size (1000 sqft)')
    plt.legend()
    pPred = plt.show()
    
    #pass
    return pPred



def comparePred(f_x1, y_Train):
    data = {
        'f_x1' : f_x1,
        'y_Train' : y_Train
    }
    data = pd.DataFrame(data)
    print(data)



#MAIN PROG
# get a row from our training data
#xTrain = xTrain[0,:]
iters = 10000
alpha = 0.003
w1 = np.arange(1, n+1)#OR (1, n+1)
b1 = 500
f_x = linRegModel(xTrain, w1, b1)
#print(np.array(f_x).shape)

cstFxn = costFunction(f_x, yTrain)#or (xTrain, yTrain, w, b)
#print(cstFxn)

j_dw, j_db = derivatives(f_x, xTrain, yTrain)
#print(j_dw, j_db)

w, b = gardientDescent(w1, b1, alpha, j_dw, j_db, iters)
print(f'w: {w}\nb: {b}')

#input('Press Enter to Predict...')
f_x1 = linRegModel(xTrain, w, b)
#print(len(f_x1))
#print(len(yTrain))

data = comparePred(f_x1, yTrain)

pData = plotData(f_x1, yTrain)
pPred = plotPred(f_x1, yTrain, w1)

print(pData)

print(pPred)
