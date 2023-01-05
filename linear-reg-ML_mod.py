# Author Dr. M. Alwarawrah
import math, os, time, scipy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score

# start recording time
t_initial = time.time()

#Columns names
col_names = ["Make","Model","Vehicle_Class","Engine_Size","Cylinders","Transmission","Fuel_Type","Fuel_Consumption_City","Fuel_Consumption_Hwy","Fuel_Consumption_Comb","Fuel_Consumption_Comb_mpg","CO2_Emissions"]
#Read dataframe and skip first raw that contain header
df = pd.read_csv('CO2 Emissions_Canada.csv',names=col_names, header = None, skiprows = 1)

colname_x = "Engine_Size"
colname_y = "CO2_Emissions"
cdf = df[[colname_x, colname_y]]

#draw histograms for the following features
plt.clf()
cdf.hist()
plt.savefig("hist.png")

# plot scatter plots
plt.clf()
fig, ax = plt.subplots()
ax.scatter(df[[colname_x]], df[[colname_y]],  color='k')
ax.set_xlabel("%s"%colname_x.replace("_"," ")) 
ax.set_ylabel("%s"%colname_y.replace("_"," "))
plt.savefig("scatter.png")

#select data for train and test
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

# plot scatter plots and show train (blue) and test data (red)
plt.clf()
fig, ax = plt.subplots(1,2)
ax[0].scatter(train[[colname_x]], train[[colname_y]],  color='b', label='Train', marker="o", s=10)
ax[0].set_xlabel("%s"%colname_x.replace("_"," ")) 
ax[0].set_ylabel("%s"%colname_y.replace("_"," "))
ax[0].legend(loc='best',frameon=False,fontsize = "8")
ax[1].scatter(test[[colname_x]], test[[colname_y]],  color='r', label='Test', marker="o", s=10)
ax[1].set_xlabel("%s"%colname_x.replace("_"," ")) 
ax[1].set_ylabel("%s"%colname_y.replace("_"," "))
ax[1].legend(loc='best',frameon=False,fontsize = "8")
plt.tight_layout()
plt.savefig("scatter_train_test.png")

#create a file to write fit information and regression accuracy
output_file = open('linear_reg_output.txt','w')

# This function use apply ML with linear regression
# You need to provide the train and test data set, columns names, and outfile name 
def linear_reg_ML(train, test, colname_x, colname_y, filename):
    #define linear regression
    regr = linear_model.LinearRegression()
    #define train data for x and y
    train_x = np.asanyarray(train[['%s'%colname_x]])
    train_y = np.asanyarray(train[['%s'%colname_y]])
    # apply the linear regression to the x & y train data
    regr.fit(train_x, train_y)
    # print columns names
    print ('%s vs. %s'%(colname_y.replace('_',' '), colname_x.replace('_',' ')), file=filename)

    # The coefficients
    m = regr.coef_[0][0]
    b = regr.intercept_[0]
    print ('Coefficients: ', m, file=filename)
    print ('Intercept: ', b, file=filename)
    #define the test data for x and y
    test_x = np.asanyarray(test[['%s'%colname_x]])
    test_y = np.asanyarray(test[['%s'%colname_y]])
    # find the prediction for y
    test_y_predict = regr.predict(test_x)
    # Regression accuracy
    MAE = np.mean(np.absolute(test_y-test_y_predict))
    print("Mean Absolute Error (MAE): %.2f" % MAE, file=filename)
    MSE = np.mean((test_y-test_y_predict) ** 2)
    print("Mean Square Error (MSE): %.2f" % MSE, file=filename)
    RMSE = np.sqrt(MSE)
    print("Root Mean Square Error (RMSE): %.2f" % RMSE, file=filename)
    RAE = np.sum(np.absolute(test_y-test_y_predict))/np.sum(np.absolute(test_y-np.mean(test_y)))
    print("Relative Absolute Error (RAE): %.2f" % RAE, file=filename)
    RSE = np.sum((test_y-test_y_predict)**2)/np.sum((test_y-np.mean(test_y))**2)
    print("Relative Square Error (RSE): %.2f" % RSE, file=filename)
    R2 = r2_score(test_y , test_y_predict) 
    print("R2-score: %.2f" %R2, file=filename)
    return train_x, m, b , R2

plt.clf()
fig, ax = plt.subplots()
# train your linear model for ENGINESIZE and CO2EMISSIONS
train_x, m, b, r2 =  linear_reg_ML(train, test, colname_x, colname_y, output_file)
ax.scatter(train[[colname_x]], train[[colname_y]],  color='blue')
ax.plot(train_x, m*train_x + b, '-r', label='R2-score: %.2f'%r2)
ax.set_xlabel("%s"%colname_x.replace("_"," ")) 
ax.set_ylabel("%s"%colname_y.replace("_"," "))
ax.legend(loc='best',frameon=False,fontsize = "8")
plt.savefig("scatter_with_linear_reg.png")
output_file.close()

#End recording time
t_final = time.time()

t_elapsed = t_final - t_initial
hour = int(t_elapsed/(60.0*60.0))
minute = int(t_elapsed%(60.0*60.0)/(60.0))
second = t_elapsed%(60.0)
print("%d h: %d min: %f s"%(hour,minute,second))
