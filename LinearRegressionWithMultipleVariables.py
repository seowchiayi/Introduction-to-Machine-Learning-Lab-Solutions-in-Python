import pandas as pd
import numpy as np
from numpy.linalg import matrix_power
import matplotlib.pyplot as plt

data = pd.read_csv('ex1data2.txt', sep=",",header=None)
data.columns = ['size of house','number of bedrooms','house price']

#keep a copy of non-normalized data to use in normal equation later
normeq = data.copy()
normeq = np.asmatrix(normeq)

# apply mean normalization which is subtracting each column values with their column mean and
# divide with column standard deviation
# DO NOT NORMALIZE HOUSE PRICE! because it is ground truth label
m_size = data['size of house'].mean()
std_size = data['size of house'].std()

m_bed = data['number of bedrooms'].mean()
std_bed = data['number of bedrooms'].std()

data['size of house'] = (data['size of house'] - m_size)/std_size
data['number of bedrooms'] = (data['number of bedrooms'] - m_bed)/std_bed

data_norm = np.asmatrix(data)
X = data_norm[:,:2]
Y = data_norm[:,2]

# add 1's column to X features
add_one = np.ones((47,1))
X = np.hstack((add_one,X))
theta = np.zeros((3, 1))
dim = X.shape
m = dim[0]


def h(X, theta):
    return X*theta


def J(X, theta, Y):
    return np.sum(np.power((h(X, theta)-Y), 2))/(2*m)


# visualizing cost curves given 4 learning rates
alpha = [['r',0.3], ['g',0.1], ['b',0.03], ['y',0.01]]
J_vals = []
_, ax = plt.subplots()
for color, a in alpha:
    print([color,a])
    for i in range(50):
        theta -= (a/m) * np.transpose(np.sum(np.multiply((h(X,theta)-Y), X),axis=0))
        J_vals.append(J(X, theta, Y))
    ax.plot(range(50), J_vals,color=color)
    J_vals=[]
    theta = np.zeros((3, 1))

ax.set(xlabel='Number of iterations', ylabel='Cost J')
plt.show()

# conclusion: learning rate of 0.3 is the best because the learning rate decreased gradually
for i in range(50):
    theta -= (0.3/ m) * np.transpose(np.sum(np.multiply((h(X, theta) - Y), X), axis=0))

print('theta found using gradient descent: ')
print(theta)
x_pred = [1,(1650-m_size)/std_size,(3-m_bed)/std_bed]
x_pred = np.asmatrix(x_pred)
ans = x_pred*theta
disp = ans.item(0)
print('house price predicted using gradient descent: ')
print(disp)


# normal equation (remember the ^-1 in the equation means the inverse of the matrix
# normal equation find theta
def normal_eq(X, Y):
    return (matrix_power((np.transpose(X)*X),-1))*(np.transpose(X)*Y)


# predict house price using normal equation
t = [1,1650,3]
t = np.asmatrix(t)
norm_theta = normal_eq(np.hstack((add_one,normeq[:,:2])),normeq[:,2])
ans1 = t*norm_theta
disp1 = ans1.item(0)
print('theta found using normal equation: ')
print(norm_theta)
print('house price predicted using normal equation: ')
print(disp1)




