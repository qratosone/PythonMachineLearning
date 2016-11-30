from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_boston
import numpy as np

boston=load_boston()
X=boston.data
y=boston.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=33)

print "max/min/avg target value: ",np.max(boston.target)," ",np.min(boston.target)," ",np.mean(boston.target)

from sklearn.preprocessing import StandardScaler
ss_X=StandardScaler()
ss_y=StandardScaler()

X_train=ss_X.fit_transform(X_train)
y_train=ss_y.fit_transform(y_train)
X_test=ss_X.transform(X_test)
y_test=ss_y.transform(y_test)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()

lr.fit(X_train,y_train)
lr_y_predict=lr.predict(X_test)

from sklearn.linear_model import SGDRegressor
sgdr=SGDRegressor()
sgdr.fit(X_train,y_train)
sgdr_y_predict=sgdr.predict(X_test)

print "lr_score:",lr.score(X_test,y_test)

from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

print "R-square:",r2_score(y_test,lr_y_predict),r2_score(y_test,sgdr_y_predict)

print "Mean_square:",mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(lr_y_predict)),mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(sgdr_y_predict))

print "Meas_absolute:",mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(lr_y_predict)),mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(sgdr_y_predict))
