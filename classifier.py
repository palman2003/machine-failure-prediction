import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve,f1_score, precision_score, recall_score
data=pd.read_csv('/kaggle/input/aircraft-engine-data/machine failure.csv')
y = data['Machine failure']
data.drop(['UDI','TWF',"HDF","PWF","OSF","RNF","Product ID","Type","Machine failure"],inplace=True,axis=1)
x=data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1,test_size=0.3)
x_train.head()
from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()


train_minmax = minmax.fit_transform(x)

X = pd.DataFrame(train_minmax, columns=x_train.columns)
X.head()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1,test_size=0.3)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,plot_confusion_matrix
rfc=RandomForestClassifier()
rfc.fit(X_train,y_train)
minmax = MinMaxScaler()
train_minmax = minmax.fit_transform(pred)
pred = pd.DataFrame(train_minmax, columns=x_test.columns)
p=rfc.predict(p)
accscore=accuracy_score(y_test,p)
print(accscore)
cm = metrics.confusion_matrix(y_test, p)
sns.heatmap(cm,annot=True,fmt='g')
#roc_auc_score
model_roc_auc = roc_auc_score(y_test, p) 
print("Area under curve:", model_roc_auc,"\n")
