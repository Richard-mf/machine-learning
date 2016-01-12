#download and view
import urllib2
url = 'http://aima.cs.berkeley.edu/data/iris.csv'
u = urllib2.urlopen(url)
localFile = open('iris.csv','w')
localFile.write(u.read())
localFile.close()

 # data easy get by numpy tools: genformtxt
from numpy import genfromtxt,zeros
#read the firt 4 colums
data = genfromtxt('iris.csv',delimiter=',',usecols=(0,1,2,3))
#read the fifth column
target = genfromtxt('iris.csv',delimiter=',',usecols=(4),dtype=str)

# plot the data
from pylab import plot,show
plot(data[target=='setosa',0],data[target=='setosa',2],'bo')
plot(data[target=='versicolor',0],data[target=='versicolor',2],'ro')
plot(data[target=='virginica',0],data[target=='virginica',2],'go')
show()

# 绘制直方图
from pylab import figure,subplot,hist,xlim,show
xmin = min(data[:,0])
xmax = max(data[:,0])
figure
subplot(411)#distribution of the setosa class(1st, on the top)
hist(data[target=='setosa',0],color='b',alpha=.7)
xlim(xmin,xmax)
subplot(412)
hist(data[target=='versicolor',0],color='r',alpha=.7)
xlim(xmin,xmax)
subplot(413)
hist(data[target=='virginica',0],color='g',alpha=.7)
xlim(xmin,xmax)
subplot(414)
hist(data[:,0],color='y',alpha=.7)
xlim(xmin,xmax)
show()

#data classified
t = zeros(len(target))
t[target == 'setosa'] =1 
t[target == 'versicolor'] = 2
t[target == 'virginica'] = 3

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(data,t)

print classifier.predict(data[0])


from sklearn import cross_validation
train,test,t_train,t_test = cross_validation.train_test_split(data,t,test_size=0.4,random_state=0)
print classifier.score(test,t_test)


#confusion_matrix

from sklearn.metrics import confusion_matrix
print confusion_matrix(classifier.predict(test),t_test)

from sklearn.metrics import classification_report
print classification_report(classifier.predict(test),t_test,target_names=['setosa','versicolor','virginical'])


from sklearn.cross_validation import cross_val_score
# cross validation with 6 iterations
scores = cross_val_score(classifier,data,t,cv=6)
print scores

from numpy import mean
print mean(scores)

#聚类
from sklearn.cluster import KMeans
kmeans = KMeans(k=3,init='random') #inistialization
kmeans.fit(data)

c=kmeans.predict(data)

from sklearn.metrics import completeness_score,homogeneity_score
print completeness_score(t,c)
print homogeneity_score(t,c)

# ???????????????? have problem
figure()

subplot(211)
plot(data[t==1,0],data[t==1,2],'bo')
plot(data[t==2,0],data[t==2,2],'ro')
plot(data[t==3,0],data[t==3,2],'go')
subplot(212)
plot(data[c==1,0],data[t==1,2],'bo',alpha=.7)
plot(data[c==2,0],data[t==2,2],'go',alpha=.7)
plot(data[c==3,0],data[t==3,2],'mo',alpha=.7)
show()

# regression
from numpy.random import rand
x=rand(40,1)
y=x*x*x+rand(40,1)/5


from sklearn.linear_model import LinearRegression
linreg=LinearRegression()
linreg.fit(x,y)
from numpy import linspace,matrix
xx=linspace(0,1,40)
plot(x,y,'o',xx,linreg.predict(matrix(xx).T),'-r')
show()

from sklearn.metrics import mean_squared_error
print mean_squared_error(linreg.predict(x),y)

