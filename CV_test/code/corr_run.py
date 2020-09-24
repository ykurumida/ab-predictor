import numpy as np
import pandas as pd
from scipy import stats
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
CONS4 = ['SIE-Scwrlmut', 'Rosmut', 'FoldXB', 'FoldXS', 'EXPT']
CONS3 = ['SIE-Scwrlmut', 'Rosmut', 'FoldXS', 'EXPT']
FI4 = ['SIE-Scwrlmut', 'Rosmut', 'DS-B', 'mCSM-AB', 'EXPT']


def standardization(df):
    y = df['EXPT']
    X = df.drop('EXPT', axis=1)
    X = (X - X.values.mean()) / X.values.std(ddof=0)
    df = pd.concat([X, y], axis=1)
    return df


def calc_gpr(df, kf):
    GP_corr_list = []
    df_gpr = df.copy()
    df_gpr = standardization(df_gpr)  # standardization
    for train, test in kf.split(df_gpr):
        kernel = (sk_kern.RBF() +
                  sk_kern.ConstantKernel() +
                  sk_kern.WhiteKernel())
        

        # separeting data
        train_df = df_gpr.iloc[train]
        test_df = df_gpr.iloc[test]
        train_X = train_df.drop('EXPT', axis=1)
        train_y = train_df['EXPT']
        test_X = test_df.drop('EXPT', axis=1)
        test_y = test_df['EXPT']

        # training
        gp_rbf = GaussianProcessRegressor(kernel=kernel)
        gp_rbf.fit(train_X, train_y)


        # Varidation
        test_gp = gp_rbf.predict(test_X)
        gp_corr = np.corrcoef(test_y, test_gp)[0, 1]
        GP_corr_list.append(gp_corr)
    return str(mean(GP_corr_list))


def calc_rf(df, kf, seed):
    Forest_corr_list = []
    df_rf = df.copy()
    df_rf = standardization(df_rf)  # standardization
    df_fi = pd.DataFrame()
    for i, data in enumerate(kf.split(df_rf)):
        train, test = data
        train_df = df_rf.iloc[train]
        test_df = df_rf.iloc[test]
        train_X = train_df.drop('EXPT', axis=1)
        train_y = train_df['EXPT']
        test_X = test_df.drop('EXPT', axis=1)
        test_y = test_df['EXPT']

        # training
        forest = RandomForestRegressor(random_state=seed)
        params = {'n_estimators': [3, 10, 100, 1000, 10000]}
        forest = GridSearchCV(forest, params, cv=4, scoring='r2', n_jobs=1,iid=False)
        y_forest = forest.fit(train_X, train_y).predict(train_X)

        # varidation
        test_forest = forest.predict(test_X)
        forest_corr = np.corrcoef(test_y, test_forest)[0, 1]
        Forest_corr_list.append(forest_corr)

        # feature importance
        df_fi[str(seed) + str(i)] = forest.best_estimator_.feature_importances_

    # feature importance
    df_fi.index = df.columns[0:-1]
    df_fi.to_csv('../Output/1/' + str(seed) + '.csv')

    return str(mean(Forest_corr_list))

def calc_mono(df, kf):
    mono_list = []
    df_mono = df.copy()
    for data in kf.split(df_mono):
        train, test = data
        test_df = df_mono.iloc[test]

        mono_corr = test_df.corr(method='pearson')
        mono_list.append(mono_corr['EXPT'])

    df_corr = pd.concat(mono_list, axis=1).mean(axis='columns')
    return df_corr


def calc_cons(df, kf):
    cons_list = []
    df_cons = df.copy()
    for data in kf.split(df_cons):
        train, test = data
        test_df = df_cons.iloc[test]
        test_X = test_df.drop('EXPT', axis=1)
        test_y = test_df['EXPT']

        test_cons = test_X.apply(calc_zscore, axis=0).mean(axis=1) 
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
    print('RFR_4\t' + calc_rf(df[FI4], kf, SEED))
    print('RFR\t' + calc_rf(df, kf, SEED))
    print('GPR\t' + calc_gpr(df_st, kf))
    print('GPR_4\t' + calc_gpr(df_st[FI4], kf))

    print(calc_mono(df, kf))


if __name__ == '__main__':
    main()
