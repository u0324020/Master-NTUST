#coding by Jane 
import numpy as np
#softmax projects vector in (0,1) interregional 
#using in one-hot encoding output a probability distribution
def softmax(x):#general
	max_score = np.max(x,axis=0) #dimension = 0
	x = x - max_score
	exp_s = np.exp(x)
	sum_exp_s = np.sum(exp_s,axis=0)
	softmax = exp_s / sum_exp_s
	return softmax

def softmax_2(x):#using numpy
	z = np.array(x)
	# a = np.exp(z)
	# b = sum(np.exp(z))
	# print(a)
	# print(b)
	# return(a/b)
	return (np.exp(z)/sum(np.exp(z)))

socres = [5.0, 1.0, 0.2]
print(softmax_2(socres))