import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math












def main():
	global xsignal
	xsignal = np.loadtxt('./xsignal.csv', delimiter=',')
	A = generateBlurMatrix(6, 3)
	print(A)


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
















