from sklearn.datasets import load_digits
digits=load_digits()
print(digits.data.shape)

from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(digits.data,digits.target,test_size=0.25,random_state=33)
print(X_train.shape)
print(X_test.shape)

from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.transform(X_test)

lsvc=LinearSVC()
lsvc.fit(X_train,Y_train)
y_predict=lsvc.predict(X_test)

print 'The Accuracy of Linear SVC is:',lsvc.score(X_test,Y_test)

from sklearn.metrics import classification_report
print classification_report(Y_test,y_predict,target_names=digits.target_names.astype(str))

