#problem one coding by Jane
#encoding = utf-8

import numpy as np

a = 10
b = 5

def ADD(ADD_a,ADD_b):
	c = np.add(ADD_a,ADD_b)
	print ("1. %d + %d = %d"%(ADD_a,ADD_b,c))

def SUBTRACT(SUBTRACT_a,SUBTRACT_b):
	c = np.subtract(SUBTRACT_a,SUBTRACT_b)
	print ("2. %d - %d = %d"%(SUBTRACT_a,SUBTRACT_b,c))

def MULTIPLY(MULTIPLY_a,MULTIPLY_b):
	c = np.multiply(MULTIPLY_a,MULTIPLY_b)
	print ("3. %d * %d = %d"%(MULTIPLY_a,MULTIPLY_b,c))

def DIVIDE(DIVIDE_a,DIVIDE_b):
	c = np.divide(DIVIDE_a,DIVIDE_b)
	print ("3. %d / %d = %d"%(DIVIDE_a,DIVIDE_b,c))

if __name__ == '__main__':
	ADD(a,b)
	SUBTRACT(a,b)
	MULTIPLY(a,b)
	DIVIDE(a,b)