import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib as mpl
import matplotlib.pyplot as plt
data_frame = pd.read_csv("weight_age_height_cloth_size.csv")


print(data_frame)
weight_value = data_frame["weight"].values.reshape(-1, 1)
height_value = data_frame["height"].values.reshape(-1, 1)

X_value = np.hstack((weight_value, height_value))
y_value = data_frame["size"].values.reshape(-1, 1)

print(X_value)
print(y_value)

X_train, X_test, y_train, y_test = train_test_split(X_value, y_value)
print(X_train.shape)
print(y_train.shape)
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

print(lin_reg.predict(X_test))
print(y_test)
print(lin_reg.score(X_train, y_train))
print(X_train)
print(X_train[:,0])
x_min = X_train[:,0].min() - 3
x_max = X_train[:,0].max() + 3

y_min = X_train[:,1].min() - 3
y_max = X_train[:,1].max() + 3

xs_plot = np.linspace(x_min, x_max, 100)
ys_plot = np.linspace(y_min, y_max, 100)

X, Y = np.meshgrid(xs_plot, ys_plot)
print(X)
print(Y)
data = pd.DataFrame(np.c_[X.ravel(), Y.ravel()], columns=["height", "size"])
print(data)
#print(data.shape)
Z = lin_reg.predict(data).reshape((100,100))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

train_plot = ax.scatter(X_train[:,0], X_train[:, 1], y_train, marker='o', label="train data")
test_plot = ax.scatter(X_test[:, 0], X_test[:, 1], y_test, marker='^', color="orange", label="test data")
#C = ax.contour3D(X, Y, Z, 50, cmap='binary')
surf = ax.plot_surface(X, Y, Z, color='red',
                       linewidth=0, antialiased=False, alpha=0.1)

ax.set_xlabel('height')
ax.set_ylabel('size')
ax.set_zlabel('weight')

fake2Dline = mpl.lines.Line2D([0],[0], linestyle="none", c='red', marker = 'o')
ax.legend([fake2Dline, train_plot, test_plot], ['model', 'train data', 'test data'], numpoints = 1)

ax.view_init(20, -40)

ax.zaxis.labelpad=-2
plt.savefig("Image.pdf")
plt.show()