from __future__ import division
import numpy as np
import math as mt
import sklearn.metrics as SkM
import matplotlib.pyplot as plt


def percentage_diff(a, b):
    ratio2 = np.zeros(len(a))

    for i in range(len(a)):
        if a[i] < b[i]:
            ratio2[i] = np.abs((b[i] / a[i]) - 1)
        else:
            ratio2[i] = (b[i] / a[i]) - 1
    return ratio2


def difference_tracker(difference, over, under, percentage):
    overestimate_groups_needed = mt.ceil(over / percentage)
    underestimate_groups_needed = mt.ceil(under / -percentage)

    overestimate_tracker = np.zeros(overestimate_groups_needed)
    underestimate_tracker = np.zeros(underestimate_groups_needed)

    print(difference)
    for i in difference:
        if i < 0:
            underestimate_tracker[mt.floor(i / -percentage)] += 1
        else:
            overestimate_tracker[mt.floor(i / percentage)] += 1

    return overestimate_tracker, underestimate_tracker


def nice_printer(difference, over, under, percentage):
    two_arrays = difference_tracker(difference, over, under, percentage)
    s0 = 0
    for val in two_arrays[0]:
        s0 += val
    print("\n" + "Overestimates: " + str(s0))
    for i in range(len(two_arrays[1])):
        print(str(i * percentage) + str("% --> ") + str(((i * percentage) + percentage)) + str("%") + "       " + str(
            two_arrays[1][i]))

    s1 = 0
    for val in two_arrays[1]:
        s1 += val
    print("\n" + "Underestimates: " + str(s1))
    for i in range(len(two_arrays[0])):
        print(str(-i * percentage) + str("% --> ") + str((-i * percentage) - percentage) + str("%") + "       " + str(
            two_arrays[0][i]))


def outputs(Y_real, Y_pred):
    percentage = 10  # defines the interval on the table



    # calculations done on data, required for the functions
    ratio = percentage_diff(Y_real, Y_pred) * 100
    minus = Y_real - Y_pred
    over = np.max(ratio)
    under = np.min(ratio)

    nice_printer(ratio, over, under, percentage)  # the interval error table

    # ERRORS
    RMSPE = np.sqrt(sum(((Y_real - Y_pred) / Y_real) ** 2) / len(Y_real))
    print("\n" + str("Root Mean Square Percentage Error:  ") + str(RMSPE))

    AvE = (sum(minus) / len(minus))
    print("\n" + str("Average Error:  ") + str(AvE))
    # is model biased toward positive or negative error

    MAE = SkM.mean_absolute_error(Y_real, Y_pred)
    print(str("Mean Absolute Error:  ") + str(MAE))
    # magnitude of error

    MedAE = np.median(np.abs(minus - np.median(minus)))
    print("\n" + str("Median Absolute Error:  ") + str(MedAE))
    print(SkM.median_absolute_error(Y_real, Y_pred))

    MSE = mt.sqrt(SkM.mean_squared_error(Y_real, Y_pred))
    print("\n" + str("Mean Squared Error:  ") + str(MSE))

    RMSE = np.sqrt(MSE)
    print(str("Root Mean Squared Error:  ") + str(RMSE))

    print("r squared value:  " + str(SkM.r2_score(Y_real, Y_pred)))

    # PLOT
    p1 = np.polyfit(Y_real, Y_pred, 1)
    plt.plot(Y_real, Y_pred, 'ro', markerfacecolor='blue', markersize=4)
    plt.plot(Y_real, np.polyval(p1, Y_real), label="Regression Line")
    plt.plot(Y_real, Y_real, color='red', linewidth=1, label="Real vs Real")
    plt.legend(loc='upper center')
    plt.show()



if __name__ == "__main__":
    '''
    This will not be run when imported to another library only if run from this one itself.
    '''
    # INPUT values
    Y_real = np.array([100, 100, 450, 550, 550, 553, 683, 688, 500, 1800, 410, 277, 305])
    Y_pred = np.array([51, 154, 530, 530, 570, 500, 444, 788, 540, 1670, 516, 317, 295])
    outputs(Y_real, Y_pred)
