import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time
# Reading Data Sets
data = pd.read_csv('Admission_Predict.csv')
datatest = pd.read_csv('Admission_Predict_Ver1.1.csv')
YactTest = datatest.iloc[400:,8] #Test dependent Values
Qt = [0, 0, 0, 0, 0, 0, 0]
for i in range(7):
    Qt[i] = datatest.iloc[400:, i+1] #Test independent Values

# HyperParameters
#m = [-0.0008910404701443437, 0.008847892432110255, 0.004944076057820275, 0.003970150305172506, 0.0036324109960603193, 0.0019574557247442146, 0.0018669544906314457]  # coefficients after 30K iteration
#c = -0.000200122395524308    # intersection points  after 30K iteration
m = [0, 0, 0, 0, 0, 0, 0]  # coefficients
c = 0  # intersection point
Q = [0, 0, 0, 0, 0, 0, 0, 0]
Qn = [0, 0, 0, 0, 0, 0, 0, 0]
D_m = [0, 0, 0, 0, 0, 0, 0]
Li = 0.000001  # Initial Learning Rate
L = Li          # Adaptive Learning Rate
iteration = 30000

for i in range(8):
    Q[i] = data.iloc[:, i + 1]  # independent training values

Q = np.array(Q)
Q = Q.reshape((8, 400)) #Reshape data list to array
rng = np.random.default_rng()
costHist =[]
iterationHist = []
minibatchsize = 50  # batch size (hyperparamater)
#print(Qn)
start = time.time()
# Performing Mini-Batch Gradient Descent
for k in range(iteration):
    rng.shuffle(Q, axis=1)  # Shuffle data set in every iteration
    if (k > 10000):  # If we start to decay earlier, the Learning rate decreases immediately, which reduces the algorithm speed.
        I = np.log(k + 1)  # For Decayin Learning rate process
        if (I == 0 or I == '-inf' or I == 'inf'):  # avoiding both divided by zero and L=0
            I = 1
        L = Li / (2 ** I)  # Geometrically decaying learning rate
    for e in range(int(400/minibatchsize)-1):
        for i in range(8):  # Just for Creating new array for calculation.
            Qn[i] = Q[i][(minibatchsize*e):(minibatchsize*(e+1))]
        Y_pred = m[0] * Qn[0] + m[1] * Qn[1] + m[2] * Qn[2] + m[3] * Qn[3] + m[4] * Qn[4] + m[5] * Qn[5] + m[6] * Qn[6] + c  # Function for predicted value
        for j in range(7):
            D_m[j] = (-1 / minibatchsize) * sum(Qn[j] * (Qn[7] - Y_pred))  # Derivatives w.r.t m's
            m[j] = m[j] - L * D_m[j]  # Update m
        D_c = (-1/minibatchsize) * sum(Qn[7] - Y_pred)  # Derivative w.r.t c
        c = c - L * D_c  # Update c
        cost = (1/minibatchsize) * sum((Qn[7]-Y_pred) ** 2)
        costHist.append(cost)
        iterationHist.append(k)
    print("cost {} , iteration {}".format(cost, k))


Y_pred = m[0] * Qt[0] + m[1] * Qt[1] + m[2] * Qt[2] + m[3] * Qt[3] + m[4] * Qt[4] + m[5] * Qt[5] + m[6] * Qt[6] + c  # Function for the predicted value in test set
SSE = sum((Y_pred - YactTest) ** 2) #Sum Squared Error  / #Variance of our linear model
MSE = SSE/100    #Mean Square Error.
RMSE = np.sqrt(MSE)             #Root Mean Square Error
YactTestMean = np.mean(YactTest)
SSR = sum((YactTestMean - YactTest) ** 2) #Total variance of the target variable
SST = SSR + SSE         # SST=SSR+SSE
rsquare = (SSR)/(SST)   # R^2 = SSR / SST
end = time.time()  #Time for calculation
print("MSE = {} R2 = {} RMSE = {} " .format(MSE,rsquare,RMSE))
print("m = {} , c = {} ".format(m,c))
print("Elapsed Time ", end - start)
plt.plot(iterationHist, costHist, 'bo')
plt.xlabel('ITERATION')
plt.ylabel('COST VALUES')
plt.title('MINI-BATCH GRADIENT DESCENT')
plt.show()
