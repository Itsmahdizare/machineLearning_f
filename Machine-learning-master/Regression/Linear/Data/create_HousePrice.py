from matplotlib import numpy as np , pyplot as plt
import pandas as pd

########################################### create the data set ###########################################
x0 = np.ones (250)
size = np.sort(np.random.choice(np.linspace(60,600,1000),250))
bedrooms = np.sort(np.random.choice(range(1,6),250,p=[.1,.35,.2,.25,.1]))
floors = np.random.choice([1,2],250)
age = np.random.choice(range(-35,0),250)

price = np.sort ( 1.8 * x0 + np.random.choice(np.linspace(4,10,25)) * size + np.random.choice(np.linspace(1.3,2.2,20)) * bedrooms + np.random.choice(np.linspace(.5,.8,20)) * floors + np.random.choice(np.linspace(3.5,4.4,20)) * age )

data = np.array(
    [
    [x0],
    [size],
    [bedrooms],
    [floors],
    [age],
    [price]
    ]
).T.reshape(250,6).astype('int32')
feature_names = np.array( [
[ 'x0','size (feet)','bedrooms','floors' , 'age (- year)' , 'price ($)']
])
data = np.concatenate((feature_names,data),axis=0)
np.savetxt('/home/mahdi/HousePrice.csv',data,fmt='%.18s',delimiter=',')

######################################################################################