import autoencoder
import math
import sys

import pandas as pd

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
	best_test_predictions =[]

	for layer1_i in range(min_layer1, max_layer1):
		for layer2_i in range(min_layer2, math.floor(layer1_i / 2) + 1):
			for encoding_size_i in range(min_encoding, layer2_i):
				validation_evaluation, validation_prediction, validation_true, test_predictions\
					= autoencoder.auto_encode(X_train, X_test, X_val, y_train, y_val, epochs=25,
											  layer1=layer1_i, layer2=layer2_i, encoding_dim=encoding_size_i,
											  batch_size=32)
				if(validation_evaluation < best_validation_evaluation):
					best_validation_evaluation = validation_evaluation
					best_validation_prediction = validation_prediction
					best_validation_true = validation_true
					best_test_predictions = test_predictions
					layer1 = layer1_i
					layer2 = layer2_i
					encoding_dim = encoding_size_i

	return layer1, layer2, encoding_dim, 32, best_validation_evaluation, \
		   best_validation_prediction, best_validation_true, best_test_predictions

def get_output(validation_prediction, validation_true, test_prediction):
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
	X_train, X_test, X_val, y_train, y_val, y_train_maximum, y_train_minimum, \
    X_train_maximum, X_train_minimum, X_test_maximum, X_test_minimum = autoencoder.prepare_data()

	layer1, encoding_dim, layer2, batch_size, validation_evaluation, \
	validation_prediction, validation_true, test_predictions = find_best_params(X_train, X_test, X_val, y_train, y_val)

	print("Best layer 1 size: " + str(layer1))
	print("Best layer 2 size: " + str(layer2))
	print("Best encoding size: " + str(encoding_dim))
	print("Validation error: " + str(validation_evaluation))
	get_output(validation_prediction, validation_true, test_predictions)
	print("Files exported")

if __name__ == '__main__':
	main()

