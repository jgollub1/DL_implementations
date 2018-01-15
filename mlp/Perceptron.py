# borrowed from CS 181, Harvard University (Spring 2016)
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as c

# You will inherit this class in your implementation of Multilayer Perceptron
class Perceptron(object):
	def visualize(self, output_file, width=3, show_charts=False, save_fig=True, include_points=True):
		X = self.X[:,1:]

		# Create a grid of points
		x_min, x_max = min(X[:, 0] - width), max(X[:, 0] + width)
		y_min, y_max = min(X[:, 1] - width), max(X[:, 1] + width)
		xx,yy = np.meshgrid(np.arange(x_min, x_max, .05), np.arange(y_min,
		    y_max, .05))

		# Flatten the grid so the values match spec for self.predict
		xx_flat = xx.flatten()
		yy_flat = yy.flatten()
		X_topredict = np.vstack((np.ones(xx_flat.shape[0]),xx_flat,yy_flat)).T

		# Get the class predictions
		Y_hat = self.predict_batch(X_topredict)
		Y_hat = Y_hat.reshape((xx.shape[0], xx.shape[1]))

		cMap = c.ListedColormap(['r','b','g'])

		# Visualize them.
		plt.figure()
		# plt.pcolormesh(xx,yy,Y_hat, cmap=cMap)
		if include_points:
			plt.scatter(X[:, 0], X[:, 1], c=self.Y, cmap=cMap)
		if save_fig:
			plt.savefig(output_file)
		if show_charts:
			plt.show()