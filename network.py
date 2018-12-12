import autoencoder
from output import outputs
from baseline import getbaseline
from sklearn.utils import shuffle

import math
import sys
import pandas as pd
import numpy as np
import csv

verbose = True


def set_verbose(x):
    """
    Sets the verbose bool to a new variable x
    :param x: the new print_progress value
    :return: Null
    """
    global verbose
    verbose = x


def maybe_print(s):
    """
    Turns off printing of checkpoints if verbose is false
    :param s: what to print
    :return: None
    """
    if verbose:
        print(s)


def read_data(file):
    """
    reads a csv file into a header array and a data array
    @author: Lasse Falch Sortland
    :param file: the filename to read
    :return: header array of type string with the header values and data array of type float with the data values
    """
    data = []
    header = []
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        counter = 0
        for row in csv_reader:
            if counter == 0:
                header = row
            else:
                i = 0
                for r in row:
                    row[i] = float(r)
                    i += 1
                data.append(row)
            counter += 1
    return header, data


def normalize(x, maximum, minimum, allow_div_by_zero=True):
    """
    Normalizes a value x based on min and max
    @author: Lasse Falch Sortland
    :param x: the value to be normalized
    :param minimum: the minimum value in the range
    :param maximum: the maximum value in the range
    :param allow_div_by_zero: if max is equal to min it returns 0 when true and throws error if false
    :return: the normalized value as float
    """
    if maximum - minimum == 0:
        if allow_div_by_zero:
            return 0
        else:
            raise TypeError('Division by zero')
    #return float(1 - ((maximum - x) / (maximum - minimum)))
    return float(((x - minimum) / (maximum - minimum)))


def denormalize(x, maximum, minimum):
    """
    Normalizes a value x based on min and max
    @author: Lasse Falch Sortland
    :param x: the value to be normalized
    :param minimum: the minimum value in the range
    :param maximum: the maximum value in the range
    :param allow_div_by_zero: if max is equal to min it returns 0 when true and throws error if false
    :return: the normalized value as float
    """
    return float(x * (maximum - minimum) + minimum)

def normalize_array(training_data):
    """
    Normalizes an array based on the max and min values of each column
    @author: Lasse Falch Sortland
    :param training_data:
    :return: return a normalized version of the input array and the max and min values
    """
    # normalize
    col = 0
    maximum = []
    minimum = []
    size = np.asarray(training_data[0]).shape[0]
    for val in range(size):
        maximum.append(max(training_data[:, val]))
        minimum.append(min(training_data[:, val]))
    for trainDat in training_data:
        row = 0
        for dat in trainDat:
            # maximum = np.max(training_data[:, row])
            # minimum = np.min(training_data[:, row])
            training_data[col][row] = normalize(float(dat), maximum[row], minimum[row])
            row += 1
        col += 1
    return training_data, maximum, minimum


def prepare_data(X_train_dir='data_files/X_train.csv', y_train_dir='data_files/y_train.csv',
                 X_test_dir='data_files/X_test.csv', validation_size=100):
    """
    Converts csv files into arrays structured and ready for use in the auto encoder
    @author Lasse Falch Sortland
    :param X_train_dir: The location of the training csv file with missing data
    :param y_train_dir: The location of the training csv file with all data
    :param X_test_dir: The location of the test csv file
    :param validation_size: the size of the validation, the bigger the validation the smaller the training data
    :return: X_train, X_test, X_val, y_train, y_val as arrays of normalized values
             y_train_maximum, y_train_minimum, X_train_maximum, X_train_minimum, X_test_maximum, X_test_minimum
             which are can be used to denormalize the arrays again
    """
    maybe_print("reading data")
    X_header, X_train = read_data(X_train_dir)
    y_header, y_train = read_data(y_train_dir)
    test_header, X_test = read_data(X_test_dir)

    maybe_print("converting array to np.array")
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)

    maybe_print("Normalize arrays")
    # Scales the training and test data to range between 0 and 1.
    y_train, y_train_maximum, y_train_minimum = normalize_array(y_train)
    maybe_print("\ty_train done")
    X_train, X_train_maximum, X_train_minimum = normalize_array(X_train)
    maybe_print("\tX_train done")
    X_test, X_test_maximum, X_test_minimum = normalize_array(X_test)
    maybe_print("\tX_test done")

    maybe_print("reshape arrays")
    # reshape the arrays
    X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
    y_train = y_train.reshape((len(y_train), np.prod(y_train.shape[1:])))
    X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))
    
    maybe_print("shuffle data")
    X_train, y_train = shuffle(X_train, y_train, random_state=0)
    
    maybe_print("create validation sets")
    # takes the last validation_size points off the training data to be used for evaluation of the algorithm
    X_val = X_train[-validation_size:]
    y_val = y_train[-validation_size:]
    # Removes the data points used for the validation array to avoid training and validating on the same data
    X_train = X_train[0:-validation_size]
    y_train = y_train[0:-validation_size]

    return X_train, X_test, X_val, y_train, y_val, y_train_maximum, y_train_minimum, \
           X_train_maximum, X_train_minimum, X_test_maximum, X_test_minimum


def find_best_params(X_train, X_test, X_val, y_train, y_val):
    """
    Finds best parameters for the autoencoder, which include layer 1, layer 2, and encoding dimension
    sizes and batch size according to their loss. Returns the optimal parameters along with the test
    results of the best model.
    :param X_train: the unlabeled training data as arr
    :param X_test: the unlabeled testing daata as arr
    :param X_val: the unlabeled validation data as arr
    :param y_train: the labeled training data as arr
    :param y_val: the labeled validation data as arr
    :return: layer1: the optimal size of layer one
             layer2: the optimal size of layer two
             encoding_dim: the optimal size of the encoding layer
             batch_size: the best batch size for the training
             validation_evaluation: the Scalar test loss of the best model
             validation_prediction: the predicted values for the best validation array
             validation_true: the actual values for the best validation array
             test_predictions: the predicted values for the best test array
    """
    min_layer1 = 4
    max_layer1 = len(X_train)
    min_layer2 = 2
    min_encoding = 1

    layer1 = min_layer1
    layer2 = min_layer2
    encoding_dim = min_encoding
    best_validation_evaluation = sys.maxsize
    best_validation_prediction = []
    best_validation_true = []
    best_test_predictions = []

    for layer1_i in range(min_layer1, max_layer1):
        for layer2_i in range(min_layer2, math.floor(layer1_i / 2) + 1):
            for encoding_size_i in range(min_encoding, layer2_i):
                validation_evaluation, validation_prediction, validation_true, test_predictions \
                    = autoencoder.auto_encode(X_train, X_test, X_val, y_train, y_val, epochs=25,
                                              layer1=layer1_i, layer2=layer2_i, encoding_dim=encoding_size_i,
                                              batch_size=32)
                if (validation_evaluation < best_validation_evaluation):
                    best_validation_evaluation = validation_evaluation
                    best_validation_prediction = validation_prediction
                    best_validation_true = validation_true
                    best_test_predictions = test_predictions
                    layer1 = layer1_i
                    layer2 = layer2_i
                    encoding_dim = encoding_size_i

    return layer1, layer2, encoding_dim, 32, best_validation_evaluation, \
           best_validation_prediction, best_validation_true, best_test_predictions


def get_output_files(validation_prediction, validation_true, test_prediction):
    validation_prediction_output = [[features[0], features[3]] for features in validation_prediction]
    validation_true_output = [[features[0], features[3]] for features in validation_true]
    test_prediction_output = [[features[0], features[3]] for features in test_prediction]

    header = ["id", "sales"]
    validation_prediction_dataframe = pd.DataFrame(validation_prediction_output, columns=header).groupby(["id"]).sum()
    validation_true_dataframe = pd.DataFrame(validation_true_output, columns=header).groupby(["id"]).sum()
    test_prediction_dataframe = pd.DataFrame(test_prediction_output, columns=header).groupby(["id"]).sum()

    validation_prediction_dataframe.to_csv("validation_prediction.csv")
    validation_true_dataframe.to_csv("validation_true.csv")
    test_prediction_dataframe.to_csv("test_output.csv")


def main():
    is_find_best_params = False
    X_train, X_test, X_val, y_train, y_val, y_train_maximum, y_train_minimum, \
    X_train_maximum, X_train_minimum, X_test_maximum, X_test_minimum = prepare_data()

    if (is_find_best_params):
        layer1, encoding_dim, layer2, batch_size, validation_evaluation, validation_prediction, \
        validation_true, test_predictions = find_best_params(X_train, X_test, X_val, y_train, y_val)

        print("Best layer 1 size: " + str(layer1))
        print("Best layer 2 size: " + str(layer2))
        print("Best encoding size: " + str(encoding_dim))
        print("Validation error: " + str(validation_evaluation))
        get_output_files(validation_prediction, validation_true, test_predictions)
        print("Files exported")
    else:
        validation_evaluation, validation_prediction, validation_true, test_predictions \
            = autoencoder.auto_encode(X_train, X_test, X_val, y_train, y_val, epochs=1)

        i = 0
        maximum = y_train_maximum[3]
        minimum = y_train_minimum[3]
        sales = test_predictions[:, 3]
        tmp = np.zeros(np.asarray(sales).size)
        id = np.zeros(np.asarray(sales).size)
        for b in sales:
            f = denormalize(b, maximum, minimum)
            tmp[i] = int(denormalize(b, maximum, minimum))
            id[i] = int(i+1)
            i += 1
        sales = tmp

        test3 = [np.asarray(id, dtype=np.int), np.asarray(sales, dtype=np.int)]
        test3 = np.transpose(test3)
        test3 = np.array(test3, dtype=np.int)
        header = ["id", "sales"]
        test_prediction_output = test3
        #print(test3)
        test_prediction_dataframe = pd.DataFrame(test_prediction_output, columns=header).groupby(["id"]).sum()
        test_prediction_dataframe.to_csv("test_output.csv")




        # get_output_files(validation_prediction, validation_true, test)
        val_pred = validation_prediction[:, 3]
        val_true = validation_true[:, 3]



        i = 0
        maximum = y_train_maximum[2]
        minimum = y_train_minimum[2]
        for b in val_pred:
            val_pred[i] = denormalize(b, maximum, minimum)
            i += 1

        i = 0
        for b in val_true:
            val_true[i] = denormalize(b, maximum, minimum)
            i += 1



        print("#######################")
        print("#### Auto Encoder #####")
        print("#######################")
        outputs(val_pred, val_true)


        base_val_pred = getbaseline(y_val, X_val)
        #val_pred = base_val_pred[:, 3]

        # i = 0
        # for b in base_val_pred:
        #     base_val_pred[i] = normalize(b, max(base_val_pred), min(base_val_pred))
        #     i += 1
        print("###################")
        print("#### Baseline #####")
        print("###################")
        
        outputs(base_val_pred, val_true)

        print()
        print(val_pred)
        print()
        print(val_true)
        print()
        print(base_val_pred)
        print()


if __name__ == '__main__':
    main()
