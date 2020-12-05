import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time
# Reading Data Sets
data = pd.read_csv('Admission_Predict.csv')
datatest = pd.read_csv('Admission_Predict_Ver1.1.csv')
YactTest  = datatest.iloc[400:,8] #Test dependent Values
Qt = [0, 0, 0, 0, 0, 0, 0]
for i in range(7):
    Qt[i] = datatest.iloc[400:, i+1] #Test independent Values


# HyperParameters
#m = [-0.0012570737661065107, 0.0061036633858139846, 0.030689335101688586, 0.020064278194125762, 0.02642744381579843, 0.023661484440317817, 0.024386184571284206]   # coefficients after 30K iteration
#c = -0.0029299115556736875   # intersection point  after 30K iteration
m = [0, 0, 0, 0, 0, 0, 0]  # coefficients
c = 0  # intersection point
Q = [0, 0, 0, 0, 0, 0, 0, 0]
D_m = [0, 0, 0, 0, 0, 0, 0]
Li = 0.000001  # Initial Learning Rate
L = Li          # Adaptive Learning Rate
iteration = 40000

for i in range(8):
    Q[i] = data.iloc[:, i + 1]  # independent training values

Q = np.array(Q)
Q = Q.reshape((8, 400)) #Reshape data list to array
rng = np.random.default_rng()
costHist =[]
iterationHist = []
start = time.time()
# Performing Sthocastic Gradient Descent on Test Values
for k in range(iteration):
    rng.shuffle(Q, axis=1) #Shuffle data set in every iteration
    if (k>5000): # If we start to decay earlier, the Learning rate decreases immediately, which reduces the algorithm speed.
        I = np.log(k + 1) #For Decayin Learning rate process
        if (I == 0 or I == '-inf' or I == 'inf'): # avoiding both divided by zero and L=0
            I = 1
        L = Li / (2 ** I)  # Geometrically decaying learning rate
    for i in range(399):
        Y_pred = m[0] * Q[0][i] + m[1] * Q[1][i] + m[2] * Q[2][i] + m[3] * Q[3][i] + m[4] * Q[4][i] + m[5] * Q[5][i] + \
                 m[6] * Q[6][i] + c  # Function for predicted value
        for j in range(7):
            D_m[j] = -Q[j][i] * (Q[7][i] - Y_pred)  # Derivatives w.r.t m's
            m[j] = m[j] - L * D_m[j]  # Update Equation for m values
        cost = ((Q[7][i] - Y_pred) ** 2) # Cost Function
        costHist.append(cost)
        iterationHist.append(k)
        D_c = -(Q[7][i] - Y_pred)  # Derivative w.r.t c
        c = c - L * D_c  # Update Equation for c
    print("cost {} , iteration {}".format(cost, k))
end = time.time()  #End of Calculation Time

#Test Performance of Algorithm
Y_pred = m[0] * Qt[0] + m[1] * Qt[1] + m[2] * Qt[2] + m[3] * Qt[3] + m[4] * Qt[4] + m[5] * Qt[5] + m[6] * Qt[6] + c  # Function for the predicted value in test set
SSE = sum((Y_pred - YactTest) ** 2) #Sum Squared Error  / #Variance of our linear model
MSE = SSE/100  #Mean Square Error.
RMSE = np.sqrt(MSE)             #Root Mean Square Error
YactTestMean = np.mean(YactTest)
SSR = sum((YactTestMean - YactTest) ** 2) #Total variance of the target variable
SST = SSR + SSE         # SST=SSR+SSE
rsquare = (SSR)/(SST)   # R^2 = SSR / SST

print ("MSE = {} R2 = {} RMSE = {} " .format(MSE,rsquare,RMSE))
print("m = {} , c = {} ".format(m,c))
print("Elapsed Time ", end - start)
plt.plot(iterationHist, costHist, 'bo')
plt.xlabel('ITERATION')
plt.ylabel('COST VALUES')
plt.title('STOCHASTIC GRADIENT DESCENT')
plt.show()

