import numpy as np
import math as mt
import sklearn.metrics as SkM

from scipy import stats

#import seaborn as sns
#sns.set(color_codes=True)
#import pandas as pd
#import matplotlib.pyplot as plt
#import statsmodels.api as sm
#from scipy.interpolate import *


def outputs(Y_real, Y_pred):
    minus = Y_real - Y_pred
    nice_printer(minus)   
 
    AvE = (sum(minus)/len(minus))
    print ("\n"+str("Average Error:  ") + str(AvE))
    # is model biased toward positive or negative error
    
    MAE = SkM.mean_absolute_error(Y_real, Y_pred)
    print (str("Mean Absolute Error:  ") + str(MAE))
    # magnitude of error
    
    MedAE  =np.median(np.abs(minus-np.median(minus)))
    print ("\n"+str("Median Absolute Error:  ") + str(MedAE))
    print (SkM.median_absolute_error(Y_real,Y_pred))
    
    MSE = mt.sqrt(SkM.mean_squared_error(Y_real, Y_pred))
    print ("\n"+str("Mean Squared Error:  ") + str(MSE))
    
    RMSE = np.sqrt(MSE)
    print (str("Root Mean Squared Error:  ")+ str(RMSE))
    
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(Y_real,Y_pred)
    print ("\n"+"r value:", r_value)
    print ("r squared value:", r_value**2)
    print (SkM.r2_score(Y_real,Y_pred))


def difference_tracker(difference):
    
    overestimate_groups_needed = mt.ceil(np.max(difference)/100)+1

    underestimate_groups_needed = mt.ceil(np.min(difference)/-100)+1

    overestimate_tracker = np.zeros(overestimate_groups_needed)
    underestimate_tracker = np.zeros(underestimate_groups_needed)
    
    print(difference)
    for i in difference:
        if i<0:
            underestimate_tracker[mt.floor(i/-100)]+=1
        else:
            overestimate_tracker[mt.floor(i/100)]+=1
            
    return overestimate_tracker,underestimate_tracker
    

def nice_printer(difference):
    two_arrays = difference_tracker(difference)
    print("\n"+"Overestimates: " +str(len(two_arrays[0])))
    for i in range(len(two_arrays[0])):
        print(str(i*100)+str(" --> ")+str((i+1)*100)+"       "+str(two_arrays[0][i]))
    
    print("\n"+"Underestimates: " +str(len(two_arrays[1])))
    for i in range(len(two_arrays[1])):
        print(str(-i*100)+str(" --> ")+str((-i-1)*100)+"       "+str(two_arrays[1][i]))    
    

if __name__ == "__main__":
    '''
    This will not be run when imported to another library only if run from this one itself.
    '''
    Y_real=np.array([688,500,1800,-10,-77,-305])
    
    Y_pred=np.array([788,540,1670,-16,-117,-295])
    
    outputs(Y_real, Y_pred)


# p1 = np.polyfit(Y_real,Y_pred,1)
# plt.plot(Y_real, Y_pred, 'ro')
# plt.plot(Y_real, np.polyval(p1,Y_real))
# plt.show()



#g = sns.lmplot(x=Y_real, y=Y_pred,data=stats)


