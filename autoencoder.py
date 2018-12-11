from keras.models import Model, Sequential
from keras.layers import Input, Dense

import numpy as np

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
	#DEPRECATED DUE TO MOVING FUNCTIONS
	X_train, X_test, X_val, y_train, y_val, y_train_maximum, y_train_minimum, \
		X_train_maximum, X_train_minimum, X_test_maximum, X_test_minimum = prepare_data()
	auto_encode(X_train, X_test, X_val, y_train, y_val, epochs=1)


if __name__ == "__main__":
	'''
	This will not be run when imported to another library only if run from this one itself.
	'''
	main()
