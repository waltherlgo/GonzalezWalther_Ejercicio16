import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sklearn.tree
data = pd.read_csv('OJ.csv')
# Remueve datos que no se van a utilizar
data = data.drop(['Store7', 'PctDiscCH','PctDiscMM'],axis=1)
# Crea un nuevo array que sera el target, 0 si MM, 1 si CH
purchasebin = np.ones(len(data), dtype=int)
ii = np.array(data['Purchase']=='MM')
purchasebin[ii] = 0
data['Target'] = purchasebin
# Borra la columna Purchase
data = data.drop(['Purchase'],axis=1)
# Crea un dataframe con los predictores
predictors = list(data.keys())
predictors.remove('Target')
predictors.remove('Unnamed: 0')
x_train,x_test,y_train,y_test= train_test_split(np.array(data[predictors]), purchasebin, train_size=0.5)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
F1trm=np.zeros(10)
F1tem=np.zeros(10)
F1trd=np.zeros(10)
F1ted=np.zeros(10)
IMm=np.zeros((10,x_train.shape[1]))
IMs=np.zeros((10,x_train.shape[1]))
Tr=100
IM=np.zeros((Tr,x_train.shape[1]))
for depth in range(1,11):
    clf = sklearn.tree.DecisionTreeClassifier(max_depth=depth)
    F1train=np.zeros(Tr)
    F1test=np.zeros(Tr)
    for i in range(Tr):
        l=np.random.randint(0,x_train.shape[0],size=x_train.shape[0])
        YT=y_train[l]
        XT=x_train[l,:]
        clf.fit(XT, YT)
        IM[i,:]=clf.feature_importances_
        F1train[i]=sklearn.metrics.f1_score(YT, clf.predict(XT))
        F1test[i]=sklearn.metrics.f1_score(y_test, clf.predict(x_test))
    F1trm[depth-1]=np.mean(F1train)
    F1tem[depth-1]=np.mean(F1test)
    F1trd[depth-1]=np.std(F1train)
    F1ted[depth-1]=np.std(F1test)
    IMm[depth-1,:]=np.mean(IM,0)
    IMs[depth-1,:]=np.std(IM,0)
plt.errorbar(np.arange(1,11),F1trm,F1trd,fmt='o')
plt.errorbar(np.arange(1,11),F1tem,F1ted,fmt='o')
plt.xlabel('max depth')
plt.ylabel('Average F1-score')
plt.legend(['train','test'])
plt.savefig('F1_training_test.png')
plt.show()
plt.figure()
Legends=[]
for i in range(14):
    stri='Col'+str(i)
    plt.plot(np.arange(1,11),IMm[:,i])
    Legends.append(stri)
plt.legend(Legends)
plt.ylabel('Average feature importance')
plt.xlabel('max depth')
plt.savefig('features.png')