'''
Regressao nao linear e ou linear
'''
import pandas as pd
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn import ensemble
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

df = pd.read_csv('parkinsons_updrs.csv')
cols = df.columns.tolist()
motor_UPDRS = cols.pop(4)
total_UPDRS = cols.pop(4)
cols.append(motor_UPDRS)
cols.append(total_UPDRS)
df=df[cols]
df = df.dropna()
y1 = df['motor_UPDRS'].values
X1 = df.loc[:,'Jitter(%)':'PPE'].values

df = pd.read_csv('CASP.csv')
df = df.dropna()
y = df['RMSD']
X = df.loc[:,'F1':'F9']

X = scale(X)
pca = PCA(n_components=6)
Xpca = pca.fit_transform(X)
X = Xpca

nfolds = 10
kf = KFold(n_splits=nfolds,shuffle=True)
params = {'n_estimators': 500,
          'max_depth': 4,
          'min_samples_split': 2,
          'learing_rate': 0.0001,
          'loss': 'ls'}

regress = {'BOOST': ensemble.GradienteBoostingRegressor(**params),
           'RF': RandomForestRegressor(n_estimators=1000, n_jobs=-1),
           'LINR': linear_model.SGDRegressor(max_iter=500, tol=1e-6),
           'MLP': MLPRegressor(hidden_layer_sizes=(100 ),max_iter=500),
           'svr_rbf': SVR(kernel='rbf',C=1e3, gamma=0.1),
           'svr_li': SVR(kernel= 'linear', C=1e3),
           'svr_poly': SVR(kernel='poly',C=1e3, degree=2)
           }

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    for name, regr in regress.items():
        #Train the model using the training sets
        regr.fit(X_train,y_train)
        y_pred = regr.predict(X_test)
        print(name)
        print('MSE: %.2f' % mean_squared_error(y_test, y_pred))
        print('R2 score: %.2f, 1 is the best! ' % r2_score(y_test,y_pred))



