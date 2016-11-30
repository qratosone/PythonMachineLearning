
#from sklearn.datasets import fetch_20newsgroups

#news = fetch_20newsgroups(subset='all')

#print len(news.data)
#print news.data[0]


from sklearn.datasets import load_iris
iris=load_iris()
print iris.data.shape

print iris.DESCR

from sklearn.cross_validation import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(iris.data,iris.target,test_size=0.25,random_state=33)

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.transform(X_test)
knc=KNeighborsClassifier()
knc.fit(X_train,Y_train)
y_predict=knc.predict(X_test)

print 'kNN Score:',knc.score(X_test,Y_test)

from sklearn.metrics import classification_report
print classification_report(Y_test,y_predict,target_names=iris.target_names)
