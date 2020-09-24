import pandas as pd
import os

# python fi_analysis.py
INPUT_DIR = '../Output/1/'
OUTPUT_DIR = '../Output/analysis/'


def main():
    if not os.path.isdir(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    features_list = []
    for i in range(1, 101):
        df_tmp = pd.read_csv(INPUT_DIR + str(i) + '.csv', index_col=0)
        features_list.append(df_tmp)
    df = pd.concat(features_list, axis=1)
    df['mean'] = df.mean(axis=1)
    print(df['mean'])
    df['mean'].to_csv(OUTPUT_DIR + 'avg.csv')


if __name__ == '__main__':
    main()
