import numpy as np
import pandas as pd
from scipy import stats
from sklearn.svm import SVR
from sklearn import datasets
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process import kernels as sk_kern
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from statistics import mean
import sys

# python corr_run.py INPUT_PATH SEED
# ex) python corr_run.py ../Input/ddG.csv 1
args = sys.argv
INPUT_PATH = args[1]
SEED = int(args[2])
NUM_SPLIT = 4
CONS4 = ['SIE-Scwrlmut', 'Rosmut', 'FoldXB', 'FoldXS', 'energy']
CONS3 = ['SIE-Scwrlmut', 'Rosmut', 'FoldXB', 'energy']
FI4 = ['SIE-Scwrlmut', 'Rosmut', 'DS_BIND', 'mCSM', 'energy']


def standardization(df):
    y = df['energy']
    X = df.drop('energy', axis=1)
    X = (X - X.values.mean()) / X.values.std(ddof=0)
    df = pd.concat([X, y], axis=1)
    return df


def standardization_all(df):
    df = (df - df.values.mean()) / df.values.std(ddof=1)
    return df


def calc_gpr(df, kf):
    GP_corr_list = []
    df_gpr = df.copy()
    df_gpr = standardization(df_gpr)  # standardization
    for train, test in kf.split(df_gpr):
        kernel = (sk_kern.RBF(1.0, (1e-3, 1e3)) +
                  sk_kern.ConstantKernel(1.0, (1e-3, 1e3)) +
                  sk_kern.WhiteKernel())

        gp_rbf = GaussianProcessRegressor(kernel=kernel)

        # separeting data
        train_df = df_gpr.iloc[train]
        test_df = df_gpr.iloc[test]
        train_X = train_df.drop('energy', axis=1)
        train_y = train_df['energy']
        test_X = test_df.drop('energy', axis=1)
        test_y = test_df['energy']

        # training
        gp_rbf.fit(train_X, train_y)

        # Varidation
        test_gp = gp_rbf.predict(test_X)
        gp_corr = np.corrcoef(test_y, test_gp)[0, 1]
        GP_corr_list.append(gp_corr)
    return str(mean(GP_corr_list))


def calc_svr(df, kf):
    SVR_corr_list = []
    df_svr = df.copy()
    df_svr = standardization(df_svr)  # standardization
    for train, test in kf.split(df_svr):
        svr_rbf = SVR(kernel='linear')

        # separeting data
        train_df = df_svr.iloc[train]
        test_df = df_svr.iloc[test]
        train_X = train_df.drop('energy', axis=1)
        train_y = train_df['energy']
        test_X = test_df.drop('energy', axis=1)
        test_y = test_df['energy']

        # training
        svr_rbf.fit(train_X, train_y)

        # Varidation
        pred_svr = svr_rbf.predict(test_X)
        svr_corr = np.corrcoef(test_y, pred_svr)[0, 1]
        SVR_corr_list.append(svr_corr)
    return str(mean(SVR_corr_list))


def calc_rf(df, kf, seed):
    Forest_corr_list = []
    df_rf = df.copy()
    df_rf = standardization(df_rf)  # standardization
    df_fi = pd.DataFrame()
    for i, data in enumerate(kf.split(df_rf)):
        train, test = data
        train_df = df_rf.iloc[train]
        test_df = df_rf.iloc[test]
        train_X = train_df.drop('energy', axis=1)
        train_y = train_df['energy']
        test_X = test_df.drop('energy', axis=1)
        test_y = test_df['energy']

        # training
        forest = RandomForestRegressor(random_state=seed)
        params = {'n_estimators': [3, 10, 100, 1000, 10000]}
        forest = GridSearchCV(forest, params, cv=4, scoring='r2', n_jobs=1)
        y_forest = forest.fit(train_X, train_y).predict(train_X)

        # varidation
        test_forest = forest.predict(test_X)
        forest_corr = np.corrcoef(test_y, test_forest)[0, 1]
        Forest_corr_list.append(forest_corr)

        # feature importance
        df_fi[str(seed) + str(i)] = forest.best_estimator_.feature_importances_
    return str(mean(Forest_corr_list))

    # feature importance
    df_fi.index = df.columns[0:-1]
    df_fi.to_csv('../Output/1/' + str(seed) + '.csv')


def calc_mono(df, kf):
    mono_list = []
    df_mono = df.copy()
    df_mono = standardization(df_mono)  # standardization
    for data in kf.split(df_mono):
        train, test = data
        test_df = df_mono.iloc[test]

        mono_corr = test_df.corr(method='pearson')
        mono_list.append(mono_corr['energy'])

    df_corr = pd.concat(mono_list, axis=1).mean(axis='columns')
    return df_corr


def calc_cons(df, kf):
    cons_list = []
    df_cons = df.copy()
    df_cons = standardization(df_cons)  # standardization
    for data in kf.split(df_cons):
        train, test = data
        test_df = df_cons.iloc[test]
        test_X = test_df.drop('energy', axis=1)
        test_y = test_df['energy']

        test_cons = test_X.apply(calc_zscore, axis=0).sum(axis=1) / 3
        cons_corr = np.corrcoef(test_y, test_cons)[0, 1]
        cons_list.append(cons_corr)
    return str(mean(cons_list))


def calc_zscore(score_list):
    MAD = score_list.mad()
    med = score_list.median()
    zscore = (score_list - med)/(1.4826*MAD)
    return zscore


def main():
    df = pd.read_csv(INPUT_PATH)
    kf = KFold(n_splits=NUM_SPLIT, random_state=SEED, shuffle=True)
    print('cons4\t' + calc_cons(df[CONS4], kf))
    print('cons3\t' + calc_cons(df[CONS3], kf))
    df_st = standardization(df)
    print('RFR\t' + calc_rf(df, kf, SEED))
    print('RFR_4\t' + calc_rf(df[FI4], kf, SEED))
    print('GPR\t' + calc_gpr(df_st, kf))
    print('GPR_4\t' + calc_gpr(df_st[FI4], kf))
    print('SVR\t' + calc_svr(df_st, kf))
    print('SVR_4\t' + calc_svr(df_st[FI4], kf))

    print(calc_mono(df, kf))


if __name__ == '__main__':
    main()