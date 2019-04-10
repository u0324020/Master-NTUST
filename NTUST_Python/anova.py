import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()


Input_data_nosc = np.genfromtxt("C:/Users/WNEC/Desktop/11月進度/1102/4. 1w+3k_data.csv", delimiter=',',dtype="float64")  # x

Input_data1 = sp.genfromtxt("C:/Users/WNEC/Desktop/11月進度/1102/4. 1w+3k_label.csv", delimiter=',',dtype="float64")  # y
Input_data = sc.fit(Input_data_nosc)

b = np.arange(1,Input_data.shape[1]+1)

f_classif_ , pval = f_classif(Input_data, Input_data1)
print(f_classif_)
print(Input_data.shape)
plt.bar(b,f_classif_)
plt.title('Anova_100', fontsize = 18)

plt.savefig('Anova_194+208+213.png', dpi=150)
plt.show()