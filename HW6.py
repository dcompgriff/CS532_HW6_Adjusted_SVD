import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import random as rand



def main():
	global xsignal
	xsignal = np.loadtxt('./xsignal.csv', delimiter=',')
	xsignal = xsignal.reshape((xsignal.size, 1))

	b0 = partb(xsignal, 0.01)
	b1 = partb(xsignal, 0.1)

def partc():
	pass

def partb(xsignal, msigma, plot=False):
	# Part b
	A = generateBlurMatrix(xsignal.shape[0], 30)
	w = np.ones((xsignal.shape[0], 1))
	for wi in range(0, w.shape[0]):
		w[wi] *= rand.normalvariate(0, sigma=msigma)
	# Calculate the blur.
	b = A.dot(xsignal) + w
	# Plots for msigma = 0.01 and 0.1.
	if plot:
		plt.plot(list(range(0, b.shape[0])), b, color='r')
		plt.plot(list(range(0, xsignal.shape[0])), xsignal, color='b')
		plt.xlabel('index')
		plt.ylabel('value')
		plt.title('Signal and Blurred Signal Plots, sigma=0.01')
		plt.show()

	return b



'''
bi = (1/k)(xi + xi-1 + xi-2 + ... + xi-k+1) + wi is a blur equation.
This equation makes a matrix A (nxn) for b = Ax + w.
'''
def generateBlurMatrix(n, k):
	A = np.zeros((n,n))

	for row in range(0, n):
		kind = row
		for writeIndex in range(kind, row - k, -1):
			if writeIndex < 0:
				break
			else:
				A[row, writeIndex] = 1

	return (1/float(k))*A



if __name__ == '__main__':
	main()
















