from keras.models import Model, Sequential
from keras.layers import Input, Dense

import numpy as np
import csv

verbose = False


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
	if maximum-minimum == 0:
		if allow_div_by_zero:
			return 0
		else:
			raise TypeError('Division by zero')
	return float(1 - ((maximum - x) / (maximum - minimum)))


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

	# takes the last validation_size points off the training data to be used for evaluation of the algorithm
	X_val = X_train[-validation_size:]
	y_val = y_train[-validation_size:]
	# Removes the data points used for the validation array to avoid training and validating on the same data
	X_train = X_train[0:-validation_size]
	y_train = y_train[0:-validation_size]

	return X_train, X_test, X_val, y_train, y_val, y_train_maximum, y_train_minimum, \
	       X_train_maximum, X_train_minimum, X_test_maximum, X_test_minimum


def auto_encode(X_train, X_test, X_val, y_train, y_val, layer1=4, encoding_dim=32,layer2=2,
                activation_in_encoder_layer_1='relu', activation_in_encoder_layer_2='relu',
                activation_in_encoder_layer_3='relu', activation_in_decoder_layer_1='relu',
                activation_in_decoder_layer_2='relu', activation_in_decoder_layer_3='sigmoid',
                optimizer='adam', loss='binary_crossentropy', epochs=10, batch_size=256):
	"""
	creates an autoencoder, trains it and uses it to predict some data
	@author: Lasse Falch Sortland
	:param X_train: the unlabeled training data as arr
	:param X_test: the unlabeled testing daata as arr
	:param X_val: the unlabeled validation data as arr
	:param y_train: the labeled training data as arr
	:param y_val: the labeled validation data as arr
	:param layer1: size of the outer layers
	:param layer2: size of the inner layers
	:param encoding_dim: how small the encoded layer is
	:param activation_in_encoder_layer_1: activation function: options = softmax, elu, softplus, softsign, relu, tanh,
			sigmoid, hard_sigmoid, exponential, linear
	:param activation_in_encoder_layer_2: activation function: options = softmax, elu, softplus, softsign, relu, tanh,
			sigmoid, hard_sigmoid, exponential, linear
	:param activation_in_encoder_layer_3: activation function: options = softmax, elu, softplus, softsign, relu, tanh,
			sigmoid, hard_sigmoid, exponential, linear
	:param activation_in_decoder_layer_1: activation function: options = softmax, elu, softplus, softsign, relu, tanh,
			sigmoid, hard_sigmoid, exponential, linear
	:param activation_in_decoder_layer_2: activation function: options = softmax, elu, softplus, softsign, relu, tanh,
			sigmoid, hard_sigmoid, exponential, linear
	:param activation_in_decoder_layer_3: activation function: options = softmax, elu, softplus, softsign, relu, tanh,
			sigmoid, hard_sigmoid, exponential, linear
	:param optimizer: the optimizer for the compiler: options = SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
	:param loss: Loss function: options mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,
			mean_squared_logarithmic_error, squared_hinge, hinge, categorical_hinge, logcosh, categorical_crossentropy,
			sparse_categorical_crossentropy, binary_crossentropy, kullback_leibler_divergence, poisson, cosine_proximity
	:param epochs: the amount of epochs to run
	:param batch_size: the size of the batches
	:return: validation_evaluation: the Scalar test loss of the model
			 validation_prediction: the predicted values for the validation array
			 validation_true: the actual values for the validation array
			 test_predictions: the predicted values for the validation array
	"""

	np.random.seed(1)
	input_dim = X_train.shape[1]

	maybe_print("creating auto encoder")
	autoencoder = Sequential()

	# Encoder Layers
	autoencoder.add(Dense(layer1 * encoding_dim, input_shape=(input_dim,), activation=activation_in_encoder_layer_1))
	autoencoder.add(Dense(layer2 * encoding_dim, activation=activation_in_encoder_layer_2))
	autoencoder.add(Dense(encoding_dim, activation=activation_in_encoder_layer_3))

	# Decoder Layers
	autoencoder.add(Dense(layer2 * encoding_dim, activation=activation_in_decoder_layer_1))
	autoencoder.add(Dense(layer1 * encoding_dim, activation=activation_in_decoder_layer_2))
	autoencoder.add(Dense(input_dim, activation=activation_in_decoder_layer_3))

	if verbose:
		autoencoder.summary()

	maybe_print("training...")
	autoencoder.compile(optimizer=optimizer, loss=loss)
	autoencoder.fit(X_train, y_train,
	                epochs=epochs,
	                batch_size=batch_size, verbose=verbose)
	maybe_print("done")

	maybe_print("predicting test data")
	test_predictions = autoencoder.predict(X_test, verbose=verbose)
	maybe_print("done")

	maybe_print("evaluating based on validation: ")
	validation_evaluation = autoencoder.evaluate(x=X_val, y=y_val, verbose=verbose)
	validation_prediction = autoencoder.predict(X_val)
	validation_true = y_val
	maybe_print(validation_evaluation)

	maybe_print("end")
	return validation_evaluation, validation_prediction, validation_true, test_predictions


def main():
	"""
	This is only used for running the program standalone. Only for testing and developing purposes.
	@author: Lasse Falch Sortland
	:return: None
	"""
	X_train, X_test, X_val, y_train, y_val, y_train_maximum, y_train_minimum, \
		X_train_maximum, X_train_minimum, X_test_maximum, X_test_minimum = prepare_data()
	auto_encode(X_train, X_test, X_val, y_train, y_val, epochs=1)


if __name__ == "__main__":
	'''
	This will not be run when imported to another library only if run from this one itself.
	'''
	main()
