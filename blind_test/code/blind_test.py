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
import os

# python corr_run.py INPUT_PATH SEED
# ex) python corr_run.py ../Input/ddG.csv 1

args = sys.argv
INPUT_TEST = args[1]
INPUT_PATH = '../Input/ddG_newSIE.csv'
SEED = int(args[2])
FI4 = ['SIE-Scwrlmut', 'Rosmut', 'DS-B', 'mCSM-AB', 'EXPT']
OUTPUT_DIR = '../Output/'
TEST_NAME = os.path.basename(INPUT_TEST.replace('.csv',''))

def standardization(df_train,df_test):
    y_train = df_train['EXPT']
    X_train = df_train.drop('EXPT', axis=1)
    X_train_2 = (X_train - X_train.values.mean()) / X_train.values.std(ddof=0)
    df_train = pd.concat([X_train_2, y_train], axis=1)
    y_test = df_test['EXPT']
    X_test = df_test.drop('EXPT', axis=1)
    X_test = (X_test - X_train.values.mean()) / X_train.values.std(ddof=0)
    df_test = pd.concat([X_test, y_test], axis=1)
    return df_train,df_test


def calc_gpr(df_train, df_test):
    df_gpr_train = df_train.copy()
    df_gpr_test = df_test.copy() 
    df_gpr_train, df_gpr_test = standardization(df_gpr_train,df_gpr_test)  # standardization
    kernel = (sk_kern.RBF() +
              sk_kern.ConstantKernel() +
              sk_kern.WhiteKernel())
    

    # separeting data
    train_X = df_gpr_train.drop('EXPT', axis=1)
    train_y = df_gpr_train['EXPT']
    test_X = df_gpr_test.drop('EXPT', axis=1)
    test_y = df_gpr_test['EXPT']

    # training
    gp_rbf = GaussianProcessRegressor(kernel=kernel)
    gp_rbf.fit(train_X, train_y)


    # Varidation
    test_gp = gp_rbf.predict(test_X)
    df_test_gp = pd.Series(test_gp)
    p_df = pd.concat([test_y,df_test_gp],axis=1)
    p_df.to_csv(os.path.join(OUTPUT_DIR,TEST_NAME+'_GPR_'+ str(SEED) + '.log'))
    gp_corr = np.corrcoef(test_y, test_gp)[0, 1]
    return str(gp_corr)


def calc_rf(df_train, df_test, seed):
    Forest_corr_list = []
    df_rf_train = df_train.copy()
    df_rf_test = df_test.copy()
    df_rf_train, df_rf_test = standardization(df_rf_train,df_rf_test)  # standardization

    train_X = df_rf_train.drop('EXPT', axis=1)
    train_y = df_rf_train['EXPT']
    test_X = df_rf_test.drop('EXPT', axis=1)
    test_y = df_rf_test['EXPT']

        # training
    forest = RandomForestRegressor(random_state=seed)
    params = {'n_estimators': [3, 10, 100, 1000, 10000]}
    forest = GridSearchCV(forest, params, cv=4, scoring='r2', n_jobs=1,iid=False)
    y_forest = forest.fit(train_X, train_y).predict(train_X)

        # varidation
    test_forest = forest.predict(test_X)
    forest_corr = np.corrcoef(test_y, test_forest)[0, 1]
    df_test_rf = pd.Series(test_forest)
    p_df = pd.concat([test_y,df_test_rf],axis=1)
    p_df.to_csv(os.path.join(OUTPUT_DIR,TEST_NAME+'_RFR_'+ str(SEED) + '.log'))



    return str(forest_corr)


def main():
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    df_train = pd.read_csv(INPUT_PATH)
    df_test = pd.read_csv(INPUT_TEST)
    print('RFR_4\t' + calc_rf(df_train[FI4],df_test[FI4],SEED))
    print('GPR_4\t' + calc_gpr(df_train[FI4],df_test[FI4]))


if __name__ == '__main__':
    main()
