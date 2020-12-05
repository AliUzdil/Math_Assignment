import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
#plt.rcParams['figure.figsize'] = (12.0, 9.0)
data = pd.read_csv('Admission_Predict.csv')
datatest = pd.read_csv('Admission_Predict_Ver1.1.csv')
# Preprocessing Input data

Yact = data.iloc[:, 8]  # Dependent Value
YactTest = datatest.iloc[400:,8] #Test dependent Values
Qt = [0, 0, 0, 0, 0, 0, 0]

for i in range(7):
    Qt[i] = datatest.iloc[400:, i+1] #Test independent Values


# data[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research']],data[['Chance of Admit ']])
#m = [-0.0024871192385996325, 0.012672645712450482, 0.012559815512679352, 0.009870135860383825, 0.009585132420421874, 0.005264206350682188, 0.0053961251876386875]   # coefficients after 40K iteration
m = [0, 0, 0, 0, 0, 0, 0]  # coefficients
Q = [0, 0, 0, 0, 0, 0, 0]
D_m = [0, 0, 0, 0, 0, 0, 0]
#c = -0.00032271985178573977     # intersection point  after 40K iteration
c = 0  # intersection point
L = 0.00000003  # Learning Rate
Li = L
iteration = 40000
tol = 0.0000001 # for break
start = time.time()
for i in range(7):
    Q[i] = data.iloc[:, i + 1]  # independent values


costHist =[]
iterationHist = []
# Performing Standart Gradient Descent
for k in range(iteration):
    if (k > 10000):  # If we start to decay earlier, the Learning rate decreases immediately, which reduces the algorithm speed.
        I = np.log(k + 1)  # For Decaying Learning rate process
        if (I == 0 or I == '-inf' or I == 'inf'):  # avoiding both divided by zero and L=0
            I = 1
        L = Li / (2 ** I)  # Geometrically decaying learning rate
    Y_pred = m[0] * Q[0] + m[1] * Q[1] + m[2] * Q[2] + m[3] * Q[3] + m[4] * Q[4] + m[5] * Q[5] + m[6] * Q[6] + c  # Function for predicted value
    # print(Y_pred)
    for j in range(7):
        D_m[j] = - sum(Q[j] * (Yact - Y_pred))  # Derivatives w.r.t m's
        #D_m[j] = -sum(Q[j] * (Yact - Y_pred))  # Derivatives w.r.t m's
        m[j] = m[j] - L * D_m[j]  # Update m
    cost = (1/2) * sum((Yact-Y_pred) ** 2)
    costHist.append(cost)
    iterationHist.append(k)
    if(k>1):
        if (cost >= (costHist[k-1] - tol)):
            print("there is no significant improvement on the cost function ")
            break
    print("cost {} , iteration {}".format(cost,k))
    D_c = -sum(Yact - Y_pred)  # Derivative w.r.t c
    c = c - L * D_c  # Update c


Y_pred = m[0] * Qt[0] + m[1] * Qt[1] + m[2] * Qt[2] + m[3] * Qt[3] + m[4] * Qt[4] + m[5] * Qt[5] + m[6] * Qt[6] + c  # Function for the predicted value in test set
SSE = sum((Y_pred - YactTest) ** 2) #Sum Squared Error  / #Variance of our linear model
MSE = SSE/100   #Mean Square Error.
RMSE = np.sqrt(MSE)             #Root Mean Square Error
YactTestMean = np.mean(YactTest)
SSR = sum((YactTestMean - YactTest) ** 2) #Total variance of the target variable
SST = SSR + SSE         # SST=SSR+SSE
rsquare = (SSR)/(SST)   # R^2 = SSR / SST
end = time.time()  #Time for calculation
print ("MSE = {} R2 = {} RMSE = {} " .format(MSE,rsquare,RMSE))
print("m = {} , c = {} ".format(m,c))
print("Elapsed Time ", end - start)
plt.plot(iterationHist, costHist, 'bo')
plt.xlabel('ITERATION')
plt.ylabel('COST VALUES')
plt.title('STANDART GRADIENT DESCENT')
plt.show()

