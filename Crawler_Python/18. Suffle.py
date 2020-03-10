import pandas as pd
import numpy as np
import random
from numpy import loadtxt

print('Read csv...')
#df1 = pd.read_csv('0224_Imbalanced_S2500.csv')
train_path = "C:/Users/Jane/Desktop/NTU/Scam/Code/0224_Imbalanced_S2500.csv"
df = loadtxt(train_path, delimiter=",")
#df2 = pd.read_csv('B_522.csv', index_col='Index')

print('shuffle csv...')
random.shuffle(df)
#df = df.sample(frac=1).reset_index(drop=True)

print('save csv...')
# df.to_csv('Test.csv')
print(df[:10])
np.savetxt('test.csv', df, delimiter=',')
print('done')
