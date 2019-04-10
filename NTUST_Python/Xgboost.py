from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import  pyplot
import warnings
import random
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
from matplotlib import pyplot as plt
from numpy import loadtxt
from sklearn.metrics import accuracy_score
import itertools
import time

warnings.filterwarnings("ignore")
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)#cm = number
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
if __name__ == '__main__':
	# create design matrix X and target vector y
	train = loadtxt("C:/Users/wnec/Desktop/11月進度/4. 1w+3k.csv", delimiter=",")
	np.random.shuffle(train)
	train_X = train[:,0:41]
	train_y = train[:,41]
	class_names =['Benign','Malicious']
	#test
	test = loadtxt("C:/Users/wnec/Desktop/12月進度/1212/992.csv", delimiter=",")
	np.random.shuffle(test)
	test_X = test[:,0:41]
	test_y = test[:,41]
	class_names =['Benign','Malicious']
	# split into train and test
	start = time.time()
	XG = XGBClassifier()
	XG.fit(train_X,train_y)
	pred = XG.predict(test_X)
	end = time.time()
	# results
	print ('TIME:',end-start)
	print ('Accuracy:',accuracy_score(test_y,pred))
	print ('Precision:',precision_score(test_y,pred))
	print ('Recall:',recall_score(test_y,pred))
	print ('F1_score:',f1_score(test_y,pred))

	cnf_matrix=confusion_matrix(test_y, pred)
	np.set_printoptions(precision=2)
	#print(cnf_matrix)
	# Plot non-normalized confusion matrix
	plt.figure()
	plot_confusion_matrix(cnf_matrix, classes=class_names,
	                    title='Confusion matrix(Xgboost), without normalization')
	plt.savefig('Xgboost1(1.3w,992new).png', dpi=150)
	# Plot normalized confusion matrix
	plt.figure()
	plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
	                    title='Normalized confusion matrix(Xgboost)')
	plt.savefig('Xgboost1-1(1.3w,992new).png', dpi=150)
	plt.show()