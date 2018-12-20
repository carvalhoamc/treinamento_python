import pandas as pd
import numpy as np
from sklearn import preprocessing, metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
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
    'MLP': MLPClassifier(max_iter=500,alpha=1)
}
nfolds = 2
kf = KFold(n_splits=nfolds,shuffle=True)
transformer = Normalizer()

train_file = './kdd/orange_small_train.data'
appetency_label_file = './kdd/orange_small_train_appetency.labels'
churn_label_file = './kdd/orange_small_train_churn.labels'
upselling_label_file = './kdd/orange_small_train_upselling.labels'

le = preprocessing.LabelEncoder()

x_df = pd.read_csv(train_file,sep='\t')
y1_df = pd.read_csv(appetency_label_file,sep='\t',header=None)
y2_df = pd.read_csv(churn_label_file,sep='\t',header=None)
y3_df = pd.read_csv(upselling_label_file,sep='\t',header=None)
x_df = x_df.fillna(0)
#verify categorical cols
cols = x_df.columns
num_cols = x_df._get_numeric_data().columns
categorical_cols = list(set(cols) - set(num_cols))
for c in categorical_cols:
    v = list(x_df[c])
    new_v = le.fit_transform(v)
    x_df[c] = new_v #new values to col dataframe


X = x_df.values
y = y1_df.values


clf = ExtraTreesClassifier(n_estimators=50)
clf = clf.fit(X, y)
model = SelectFromModel(clf, prefit=True)
X_new = model.transform(X)
X = X_new

dfcol = ['FOLD','ALGORITHM','PRE','REC','SPE','F1','GEO','IBA','ACC','AUC']

i = 0
fold = 0
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index],X[test_index]
    y_train, y_test = y[train_index],y[test_index]
    end_index = classifiers.__len__()
    idx = np.arange(0, end_index*nfolds)
    df1 = pd.DataFrame(columns=dfcol,index=idx)

    for name, clf in classifiers.items():
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        res = classification_report_imbalanced(y_test, y_pred)
        aux = res.split()
        score = aux[-7:-1]
        df1.iat[i,0] = fold
        df1.iat[i,1] = name
        df1.iat[i,2] = score[0]
        df1.iat[i, 3] = score[1]
        df1.iat[i, 4] = score[2]
        df1.iat[i, 5] = score[3]
        df1.iat[i, 6] = score[4]
        df1.iat[i, 7] = score[5]
        df1.iat[i, 8] = accuracy_score(y_test,y_pred)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
        df1.iat[i, 9] =metrics.auc(fpr, tpr)
        i = i + 1
        print(str(fold) + ' ' + str(name))
    fold = fold + 1
df1.to_csv('results_kdd.csv',index=False)

t = pd.Series(data=np.arange(0, df1.shape[0],1))
dfr = pd.DataFrame(columns=['ALGORITHM','PRE','REC','SPE','F1','GEO','IBA','ACC','AUC'],
                   index=np.arange(0, int(t.shape[0] / nfolds)))
df_temp = df1.groupby(by=['ALGORITHM'])
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
    dfr.at[i, 'AUC'] = group['AUC'].astype(float).mean()

    i = i + 1

dfr.to_csv('media_results_kdd.csv',index=False)
