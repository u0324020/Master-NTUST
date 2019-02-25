#problem three coding by Jane
#encoding = utf-8

import numpy as np

A = np.array([ 1, 3, 5, 7, 9 ])
B = np.array([ 2, 4, 6, 8, 10 ])

def ADD(ADD_a,ADD_b):
	C = []
	for i in range(0 , 5 , 1): 
		D = A[i]+B[i]
		C.append(D)
	#C = np.add(ADD_a,ADD_b) ### This's simple function
	print "ADD. "
	print " ".join([str(i) for i in ADD_a] )
	print ("+)")
	print " ".join([str(j) for j in ADD_b] )
	print ("=")
	print " ".join([str(k) for k in C] )

def MULTIPLY(MULTIPLY_a,MULTIPLY_b):
	C = []
	for i in range(0 , 5 , 1): 
		D = A[i]*B[i]
		C.append(D)
	#C = np.multiply(MULTIPLY_a,MULTIPLY_b) ### This's simple function
	print "MULTIPLY. "
	print " ".join([str(i) for i in MULTIPLY_a] )
	print ("*)")
	print " ".join([str(j) for j in MULTIPLY_b] )
	print ("=")
	print " ".join([str(k) for k in C] )

if __name__ == '__main__':
	ADD(A,B)
	MULTIPLY(A,B)