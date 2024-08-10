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


xvalue = pd.read_csv("USEconomicData2021.csv")
xvalue = xvalue["Volume of exports of goods"]
yvalue = pd.read_csv("USEconomicData2021.csv")
yvalue = yvalue["Gross domestic product, current prices"]

rawy = yvalue
rawx = xvalue
newx = rawx
newy = rawy
plt.scatter(rawx,rawy)
newx = rawx.to_numpy()
newy = rawy.to_numpy()
newx = np.reshape(newx, (-1,1))
newy = np.reshape(newy, (-1,1))

lin = plt.figure(1)
model = LinearRegression()
model.fit(newx,newy)
y_predict = model.predict(newx)
plt.plot(rawx,y_predict, color = 'red')
lin.show()
# value = model.predict([[125819]])
# plt.scatter(125819,value, color = 'yellow')
mse = mean_squared_error(newy, y_predict)
# # print("In 2015 the GDP based on it's correlation with annual house income is "+ str(value))
print("Linear")
print("Coefficents:  "+str(model.coef_))
print ("Model Accuracy Rate:  "+ str(model.score(newx, newy)))
print("Mean Squared Error:  "+ str(mse))

# # Decision Tree Regression Model
d = plt.figure(2)
plt.scatter(rawx,rawy)

model = DecisionTreeRegressor(max_depth=3).fit(newx, newy)
target_predicted = model.predict(newx)
mse = mean_squared_error(newy, target_predicted)
data = pd.DataFrame({"x": rawx, "y": rawy })

plt.plot(newx, target_predicted, color = "red")

d.show()
print("Decision Tree")
print ("Model Accuracy Rate:  "+ str(model.score(newx, newy)))
print("Mean Squared Error:  "+ str(mse))

# valuetree = tree.predict([[125000]])
# plt.scatter(125000,valuetree, color = "yellow")

## Decision Tree Regression Model



### Polynomial Regression Model
p = plt.figure(3)
plt.scatter(rawx,rawy)

model = make_pipeline(
    PolynomialFeatures(degree=3),
    LinearRegression(),
)
model.fit(newx, newy)
target_predicted = model.predict(newx)
mse = mean_squared_error(newy, target_predicted)
ax = sns.scatterplot(newx=newx,color="black", alpha=0.5)
ax.plot(newx, target_predicted, color = "red")

p.show()
print("Polynomial")
print ("Model Accuracy Rate:  "+ str(model.score(newx, newy)))
print("Mean Squared Error:  "+ str(mse))
# valuepolynomial= model.predict([[125000]])
# plt.scatter(125000,valuepolynomial,color = "red")

### Polynomial Regression Model

# ###  KBinsDiscretizer 
k = plt.figure(4)
plt.scatter(rawx,rawy)

model = make_pipeline(
    KBinsDiscretizer(n_bins=8), LinearRegression(),
)
model.fit(newx, newy)
target_predicted = model.predict(newx)
mse = mean_squared_error(newy, target_predicted)

ax = sns.scatterplot(newx = newx, color="black", alpha=0.5)
ax.plot(newx, target_predicted, color = "red")
k.show()
print("KbinsDiscretizer")
print ("Model Accuracy Rate:  "+ str(model.score(newx, newy)))
print("Mean Squared Error:  "+ str(mse))
#valueKBinsDiscretizer = model.predict([[125000]])
#plt.scatter(125000,valueKBinsDiscretizer, color = "green")
# ### KBinsDiscretizer model

##print("Coefficents:  "+str(model.coef_))
plt.show()


