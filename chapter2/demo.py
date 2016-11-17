from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import kNN
datingDataMat,datingDataLabels=kNN.file2matrix('datingTestSet2.txt')
fig=plt.figure()
ax=fig.add_subplot(111)
ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*array(datingDataLabels),15.0*array(datingDataLabels))
plt.show()