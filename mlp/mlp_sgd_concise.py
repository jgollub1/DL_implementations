import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from Perceptron import Perceptron

# to find more sample datasets
# http://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html
class MLP(Perceptron):
	def __init__(self, hidden_layer_sizes = [5]):
		self.eta = .1
		self.shape = hidden_layer_sizes
		self.num_iter = 10000

	
	# written for one hidden layer
	def fit(self, X, Y):
		self.X = X
		self.Y = Y
		self.num_inputs = X.shape[1]
		self.init_weights(X, Y)

		for i in xrange(self.num_iter):
			r_ind = np.random.choice(X.shape[0])
			z_s = self.activated_units(X[r_ind])
			delta_1 = self.predict(X[r_ind]) - Y[r_ind]
			grad_1 = z_s * delta_1
			self.gradient_update(1, grad_1)

			z_s = self.activated_units(X[r_ind])
			delta_0 = (1 - z_s**2) * (self.w[1]) * delta_1
			grad_0 = np.dot(X[r_ind].reshape(self.num_inputs,1), delta_0.T)

			self.gradient_update(0, grad_0)

	def init_weights(self, X, Y):
		self.w = [np.random.normal(0, 1, (self.num_inputs, self.shape[0]))]
		self.w.append(np.random.normal(0, 1, (self.shape[-1], 1)))		

	def gradient_update(self, w_ind, grad):
		self.w[w_ind] -= self.eta * grad


	def predict(self, X):
		z_s = self.activated_units(X)
		return sigmoid(np.dot(z_s.T, self.w[1])[0,0])

	def activated_units(self, X):
		alphas = np.dot(X, self.w[0])
		return np.tanh(alphas).reshape(alphas.shape[0],1)

	def predict_batch(self, X):
		z_s = self.activated_units_batch(X)
		return sigmoid(np.dot(z_s, self.w[1]).sum(axis=1))

	def activated_units_batch(self, X):
		alphas = np.dot(X, self.w[0])
		return np.tanh(alphas)


def sigmoid(x):
	return 1 / (1 + np.exp(-x))


if __name__=='__main__':
	# data, target = datasets.make_circles(n_samples=1500, factor=.5, noise=.05)
	# plt.scatter(data[:,0], data[:,1], c=target)
	# df = np.concatenate([data, target.reshape(len(target), 1)], axis=1)
	# df = np.concatenate([np.ones([df.shape[0], 1]), df], axis=1) # add 2D array for bias
	# np.save("circles.npy", df)

	clf = MLP()
	start = time.clock()
	df = np.load("circles.npy")
	X_train, X_test, y_train, y_test = train_test_split(df[:,0:3], df[:,3], test_size=0.3)
	
	clf.fit(X_train, y_train)
	print time.clock() - start, "seconds"

	train_preds = clf.predict_batch(X_train)
	test_preds = clf.predict_batch(X_test)
	other = clf.predict_batch(X_train[:1])
	print "train log loss:", log_loss(y_train, train_preds)
	print "train log loss:", log_loss(y_test,  test_preds)
	clf.visualize("mlp.png", width=1, show_charts=True, save_fig=True, include_points=True)

	