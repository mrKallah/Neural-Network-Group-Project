from keras.models import Model, Sequential
from keras.layers import Input, Dense

import numpy as np
import csv


def read_data(file):
	'''
	reads a csv file into a header array and a indata array
	@author: Lasse Falch Sortland
	:param file: the filename to read
	:return: header array of type string with the header values and indata array of type float with the data values
	'''
	indata = []
	header = []
	with open(file) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		couter = 0
		for row in csv_reader:
			if couter == 0:
				header = row
			else:
				i = 0
				for r in row:
					row[i] = float(r)
					i += 1
				indata.append(row)
			couter += 1
	return header, indata


def normalize(x, max, min, allowDivByZero=True):
	'''
	Normalizes a value x based on min and max
	@author: Lasse Falch Sortland
	:param x: the value to be normalized
	:param min: the minimum value in the range
	:param max: the maximum value in the range
	:param allowDivByZero: if max is equal to min it returns 0 when true and throws error if false
	:return: the normalized value as float
	'''
	if (max-min == 0):
		if (allowDivByZero):
			return 0
		else:
			raise TypeError('Division by zero')
	return float(1-((max - x) / (max - min)))


def normalizeArr(trainingData):
	'''
	Normalizes an array based on the max and min values of each column
	@author: Lasse Falch Sortland
	:param trainingData:
	:return:
	'''
	# normalize
	col = 0
	maximum = []
	minimum = []
	size = np.asarray(trainingData[0]).shape[0]
	for val in range(size):
		maximum.append(max(trainingData[:, val]))
		minimum.append(min(trainingData[:, val]))
	for trainDat in trainingData:
		row = 0
		for dat in trainDat:
			#maximum = np.max(trainingData[:, row])
			#minimum = np.min(trainingData[:, row])
			trainingData[col][row] = normalize(float(dat), maximum[row], minimum[row])
			row += 1
		col += 1
	return trainingData


def confusion_matrix_reg(true, predicted, variance=0.001):
	cm = []
	true = np.asarray(true)
	predicted = np.asarray(predicted)

	i = 0
	for row in true:
		j = 0
		cm_row = []
		for col in row:
			tmp = true[i][j] - predicted[i][j]
			if (tmp > variance):
				cm_row.append(0)
			else:
				cm_row.append(1)

			cm.append(cm_row)
			j += 1
		i += 1

	return cm

np.random.seed(1)

print("reading data")
X_header, X_train = read_data('dnnData/X_train.csv')
y_header, y_train = read_data('dnnData/y_train.csv')
test_header, X_test = read_data('dnnData/X_test.csv')


#X_train, X_test = train_test_split(indata)


print("converting array to np.array")
X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
X_test = np.asarray(X_test)

print("Normalize arrays")
# Scales the training and test data to range between 0 and 1.
y_train = normalizeArr(y_train)
print("\ty_train done")
x_train = normalizeArr(X_train)
print("\tx_train done")
x_test = normalizeArr(X_test)
print("\tx_test done")



print("reshape arrays")
# reshape the arrays
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
y_train = y_train.reshape((len(y_train), np.prod(y_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


x_val = x_train[-100:]
x_train = x_train[0:-100]

y_val = y_train[-100:]
y_train = y_train[0:-100]



# input dimension = 784
input_dim = x_train.shape[1]
encoding_dim = 32

print("")
print("creating auto encoder")
autoencoder = Sequential()


# Changeable vars for different tests
layer1 = 4
layer2 = 2
activation_in_encoder_layer_1 = 'relu'
activation_in_encoder_layer_2 = 'relu'
activation_in_encoder_layer_3 = 'relu'
activation_in_decoder_layer_1 = 'relu'
activation_in_decoder_layer_2 = 'relu'
activation_in_decoder_layer_3 = 'sigmoid'
optimizer = 'adam'
loss = 'binary_crossentropy'
epochs = 10
batch_size = 256

# Encoder Layers
autoencoder.add(Dense(layer1 * encoding_dim, input_shape=(input_dim,), activation=activation_in_encoder_layer_1))
autoencoder.add(Dense(layer2 * encoding_dim, activation=activation_in_encoder_layer_2))
autoencoder.add(Dense(encoding_dim, activation=activation_in_encoder_layer_3))

# Decoder Layers
autoencoder.add(Dense(layer2 * encoding_dim, activation=activation_in_decoder_layer_1))
autoencoder.add(Dense(layer1 * encoding_dim, activation=activation_in_decoder_layer_2))
autoencoder.add(Dense(input_dim, activation=activation_in_decoder_layer_3))

print(autoencoder.summary())


_input = Input(shape=(input_dim,))
encoder_layer1 = autoencoder.layers[0]
encoder_layer2 = autoencoder.layers[1]
encoder_layer3 = autoencoder.layers[2]
encoder = Model(_input, encoder_layer3(encoder_layer2(encoder_layer1(_input))))

print(encoder.summary())


print("training...")
autoencoder.compile(optimizer=optimizer, loss=loss)
history = autoencoder.fit(x_train,y_train,
                epochs=epochs,
                batch_size=batch_size)
print("done")

print("predicting test data")
predictions = autoencoder.predict(x_test);
print("done")

print("predictions: ")
print(predictions)
print("x_test: ")
print(x_test)

print("building confusion matrix")
cm = confusion_matrix_reg(x_test, predictions)
print("confusion matrix: ")
print(cm)
print("done")

print("evaluating based on validation: ")
print(autoencoder.evaluate(x=x_val, y=y_val))

print("end")
