import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt


#read data
dataframe = pd.read_csv('challenge_dataset.txt')
x_values = dataframe[['A']]
y_values = dataframe[['B']]

#train model on data
reg = linear_model.LinearRegression()
reg.fit(x_values, y_values)

#visualize results
plt.scatter(x_values, y_values)
plt.plot(x_values, reg.predict(x_values), 'b-')
plt.show()
