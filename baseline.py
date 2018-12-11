import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model,metrics
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import PolynomialFeatures

def getbestValue():
     #split the training set
    data = pd.read_csv('data_files/y_train.csv')
    a = data[[ 'Customers','Open','Promo','SchoolHoliday']]
    #X2 = data1[[ 'Customers','Open','Promo','SchoolHoliday']]
    b = data[['Sales']]
    x_train,x_test,y_train,y_test=train_test_split(a,b,random_state=1)
    linreg = LinearRegression() 
    linreg.fit(x_train, y_train)
    print ('b',linreg.intercept_) 
    print ('a,c,d,f',linreg.coef_) 
    array1=linreg.intercept_
    array2=linreg.coef_
      #Sales=b+a*Customers+c*Open+d*Promo"+f*SchoolHoliday
      #Can get the relationship between Sales and Customers,Open, Promo, SchoolHoliday,the coefficient
      #Sales=-95.76277426+6.1239122∗Customers+1795.55532259∗Open+1332.06787041∗Promo +106.7114884∗SchoolHoliday
      #cross validation
    y_pred = linreg.predict(x_test)
   #print(y_pred)
    predicted = cross_val_predict(linreg, a, b, cv=10)
    print ("MSE:",metrics.mean_squared_error(y_test, y_pred))
    print ("RMSE:",np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
   #write the plot
    fig, ax = plt.subplots()
    ax.scatter(b, predicted)
    ax.plot([b.min(), b.max()], [b.min(), b.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()
    #polynomial regression
    #Multiple monaadic equation
    quadratic_featurizer = PolynomialFeatures(degree=5)
    #transfer the x y from data set
    x_train_quadratic = quadratic_featurizer.fit_transform(x_train)
    x_test_quadratic = quadratic_featurizer.transform(x_test)
    #begin training
    regressor_quadratic = LinearRegression()
    regressor_quadratic.fit(x_train_quadratic, y_train)
    xx_quadratic = quadratic_featurizer.transform(x_train)
    #calcluate the R squared
    print('1 r-squared:{:.2f}%'.format (linreg.score(x_test, y_test)*100))
    print('2 r-squared:{:.2f}%'.format (regressor_quadratic.score(x_test_quadratic, y_test)*100))
    #write the plot
    yy = regressor_quadratic.predict(x_test) 
    xx_poly2 = quadratic_featurizer.transform(x_test)
    yy_poly2 = regressor_quadratic.predict(xx_poly2)
    plt.scatter(x_train, y_train)
    plt1= plt.plot(x_test, y_test)
    plt2=plt.plot(xx_poly2,yy_poly2,'k--',lw=4)
    plt.axis([0, 300000, 0, 100000])
    plt.xlabel("Diameter")
    plt.ylabel("Price")
    plt.show()
    return array1,array2

if __name__ == "__main__":
    '''
    This will not be run when imported to another library only if run from this one itself.
    '''
    a=getbestValue()



