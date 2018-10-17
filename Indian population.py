#Data analysis of India population
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import dataset
df=pd.read_csv("India_population.csv")
X=df.iloc[:, :-1]
y=df.iloc[:, 1]

#spliting dataset into two part trainingset & test set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train ,y_test =train_test_split(X, y, test_size=0.2 ,random_state=0)

#train the model using linear regression
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#predict the testset result
y_pred=regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('year vs population (Training set)')
plt.xlabel('Years')
plt.ylabel('population')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('year vs population (Test set)')
plt.xlabel('Years')
plt.ylabel('population')
plt.show()


