import pandas as pd
import numpy as np
from imblearn.metrics import classification_report_imbalanced
from sklearn import datasets
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

classifiers = {
    'RF': RandomForestClassifier(n_estimators=100),
    'KNN': KNeighborsClassifier(),
    'DTREE': DecisionTreeClassifier(),
    'GNB': GaussianNB(),
    'LRG': LogisticRegression(),
    'ABC': AdaBoostClassifier(),
    'MLP': MLPClassifier(max_iter=500,alpha=1),
    'KDA': QuadraticDiscriminantAnalysis(),
    'SVM1': SVC(kernel='linear',C=0.025),
    'SVM2': SVC(gamma=2, C=1),
    'GPC': GaussianProcessClassifier(1.0 * RBF(1.0))
}
nfolds = 10

kf = KFold(n_splits=nfolds,shuffle=True)
transformer = Normalizer()
iris = datasets.load_iris()
X = iris.data
X = transformer.fit_transform(X)
y = iris.target
dfcol = ['FOLD','ALGORITHM','PRE','REC','SPE','F1','GEO','IBA','AUC','ACC']
df = pd.DataFrame(columns=dfcol)
i = 0
fold = 0
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index],X[test_index]
    y_train, y_test = y[train_index],y[test_index]
    for name, clf in classifiers.items():
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        res = classification_report_imbalanced(y_test, y_pred)
        aux = res.split()
        score = aux[-7:-1]
        df.at[i,'FOLD'] = fold
        df.at[i,'ALGORITHM'] = name
        df.at[i,'PRE'] = score[0]
        df.at[i, 'REC'] = score[1]
        df.at[i, 'SPE'] = score[2]
        df.at[i, 'F1'] = score[3]
        df.at[i, 'GEO'] = score[4]
        df.at[i, 'IBA'] = score[5]
        df.at[i, 'ACC'] = accuracy_score(y_test,y_pred)
        i = i + 1
        print(str(fold) + ' ' + str(name))
    fold = fold + 1
df.to_csv('results_iris.csv',index=False)

t = pd.Series(data=np.arange(0, df.shape[0],1))
dfr = pd.DataFrame(columns=['ALGORITHM','PRE','REC','SPE','F1','GEO','IBA','AUC','ACC'],
                   index=np.arange(0, int(t.shape[0] / nfolds)))
df_temp = df.groupby(by=['ALGORITHM'])
idx = dfr.index.values
i = idx[0]
for name, group in df_temp:
    group = group.reset_index()
    dfr.at[i,'ALGORITHM'] = group.loc[0,'ALGORITHM']
    dfr.at[i,'PRE'] = group['PRE'].astype(float).mean()

    dfr.at[i, 'REC'] = group['REC'].astype(float).mean()
    dfr.at[i, 'SPE'] = group['SPE'].astype(float).mean()
    dfr.at[i, 'F1'] = group['F1'].astype(float).mean()
    dfr.at[i, 'GEO'] = group['GEO'].astype(float).mean()
    dfr.at[i, 'IBA'] = group['IBA'].astype(float).mean()
    dfr.at[i, 'ACC'] = group['ACC'].astype(float).mean()
    i = i + 1

dfr.to_csv('media_results_iris.csv',index=False)









