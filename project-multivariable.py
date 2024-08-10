import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
from pandas_datareader import data
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.kernel_approximation import Nystroem
from sklearn.model_selection import train_test_split


xvalue = pd.read_csv("ScienceFair Data - Sheet1.csv")
xvalue = xvalue.iloc[:,0:8]
yvalue = pd.read_csv("USEconomicData2021.csv")
yvalue = yvalue["Gross domestic product, current prices"]
print(xvalue)
rawy = yvalue
rawx = xvalue
newx = rawx
newy = rawy

x_train, x_test, y_train, y_test = train_test_split(newx,newy,test_size = 0.2)
print(x_train.shape)

# plt.scatter(newx[:,0],newy)
# plt.scatter(newx[:,1],newy)


# newx = np.reshape(newx, (-1,1))
# newy = np.reshape(newy, (-1,1))



### Data Organization


# ## Decision Tree Regression Model

tree = DecisionTreeRegressor(max_depth=3).fit(newx, newy)
target_predicted = tree.predict(newx)
msedecisiontree = mean_squared_error(newy, target_predicted)
##data = pd.DataFrame({"x": newx, "y": newy })

plt.plot(newx, target_predicted)
# valuetree = tree.predict([[116011,24418]])
# plt.scatter([0,125000]],valuetree, color = "yellow")

# ## Decision Tree Regression Model



### Polynomial Regression Model

polynomial_regression = make_pipeline(
    PolynomialFeatures(degree=3),
    LinearRegression())
polynomial_regression.fit(newx, newy)
target_predicted = polynomial_regression.predict(newx)
msepolynomial = mean_squared_error(newy, target_predicted)
ax = sns.scatterplot(newx=newx,color="black", alpha=0.5)
ax.plot(newx, target_predicted)
_ = ax.set_title(f"Mean squared error = {msepolynomial:.2f}")

# valuepolynomial= polynomial_regression.predict([[116011,24418]])
# plt.scatter(24418,valuepolynomial,color = "red")

### Polynomial Regression Model

# ###  KBinsDiscretizer or Nystroem Model

binned_regression = make_pipeline(
    KBinsDiscretizer(n_bins=8), LinearRegression(),
)
binned_regression.fit(newx, newy)
target_predicted = binned_regression.predict(newx)
msekbinsdiscretizer = mean_squared_error(newy, target_predicted)

ax = sns.scatterplot(newx = newx, color="black", alpha=0.5)
ax.plot(newx, target_predicted, color = "black")
_ = ax.set_title(f"Mean squared error = {msekbinsdiscretizer:.3f}")

# valueKBinsDiscretizer = binned_regression.predict([[116011,24418]])
##plt.scatter(125000,valueKBinsDiscretizer, color = "green")
### KBinsDiscretizer model
plt.show()