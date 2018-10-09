import matplotlib.pyplot as plt
import numpy as np

# dividing dataset into training inputs and label
data = open('ex1data1.txt','r')
X = []
Y = []
for line in data:
    x = line.split(',')
    X.append([float(x[0])])
    Y.append(float(x[1].strip()))

# visualize dataset with scatter plot using matplotlib
plt.scatter(X, Y, marker='x', color='red')
plt.xlabel('Profit in $10,000s')
plt.ylabel('Population of city in 10,000s')
plt.show()

# convert X(input), and Y(ground-truth label) into matrix because we need to find the matrix multiplication
# between X and theta
X_mat = np.asmatrix(X)
Y_mat = np.asmatrix(Y)

# add a column of ones to X_mat
add_one = np.ones((97,1))
X_mat = np.hstack((add_one,X_mat))

# initialize theta with zeros of 2 rows and 1 column
theta = np.zeros((2, 1))


# hypothesis formula which is transpose of theta multiply X
def h(X_mat, theta):
    return X_mat*theta


# get m which is the number of training examples
dim = X_mat.shape
m = dim[0]

# cost function formula which is to find the sum of squared difference between h and Y divide by 2*m
J = np.sum(np.power((h(X_mat, theta)-np.transpose(Y_mat)), 2))/(2*m)
print('Initial cost')
print(J)

# learning rate to be used in gradient descent
alpha = 0.01

# gradient descent with 1500 iterations
for i in range(1500):
    theta -= (alpha/m) * np.transpose(np.sum(np.multiply((h(X_mat, theta)-np.transpose(Y_mat)), X_mat), axis=0))

print('Final theta after gradient descent')
print(theta)

# find predict1 and predict2 as asked in ex1
a = [1,3.5]
b = [1,7]
a = np.asmatrix(a)
b = np.asmatrix(b)

predict1 = a*theta*10000
predict1 = predict1.item(0)

predict2 = b*theta*10000
predict2 = predict2.item(0)

print('for areas of 35000 we predict ' + str(predict1))
print('for areas of 70000 we predict ' + str(predict2))

# visualizing the best fit line
plt.scatter(X, Y, marker='x', color='red')
axes = plt.gca()
x_vals = np.array(axes.get_xlim())
# the equation of best fit line
y_vals = theta[0][0] + theta[1][0]* x_vals
plt.plot(x_vals, y_vals, '--')
plt.show()





