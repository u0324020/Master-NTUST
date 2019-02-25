#problem five coding by Jane
#encoding = utf-8

import numpy as np

A = np.array([[ 1, 3, 5, 7, 9 ],[ 9, 7, 5, 3, 1 ]])
B = np.array([[ 2, 4, 6, 8, 10 ],[ 10, 8, 6, 4, 2 ]])

def ADD(ADD_a,ADD_b):
	C = np.add(ADD_a,ADD_b)
	print "ADD. "
	print " ".join([str(i) for i in ADD_a] )
	print ("+)")
	print " ".join([str(j) for j in ADD_b] )
	print ("=")
	print " ".join([str(k) for k in C] )

def MULTIPLY(MULTIPLY_a,MULTIPLY_b):
	C = np.multiply(MULTIPLY_a,MULTIPLY_b)
	print "MULTIPLY. "
	print " ".join([str(i) for i in MULTIPLY_a] )
	print ("*)")
	print " ".join([str(j) for j in MULTIPLY_b] )
	print ("=")
	print " ".join([str(k) for k in C] )

if __name__ == '__main__':
	ADD(A,B)
	MULTIPLY(A,B)