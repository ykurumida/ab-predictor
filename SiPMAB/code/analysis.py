import sys
import os
import re
from statistics import mean, median, variance, stdev
import pandas as pd


INPUT_PATH = '../Output/1/log.{}'
OUTPUT_DIR = '../Output/analysis/'
DATA_NUM = 100 + 1
PREDICTOR_LIST = ['SIE-Scwrlmut', 'SIE-Rosmut', 'SIE-Rosiface-sc',
                  'SIE-RosCDR-loop', 'Rosmut', 'Rosiface-sc',
                  'RosCDR-loop', 'FoldXB', 'FoldXS', 'DS-B',
                  'DS-S', 'mCSM-AB']


def collect_data_mono(predictor_name, input_path):
    pattern = predictor_name + '\s*(-?[0-1]\.[0-9]*)$'
    corr_list = []

    for i in range(1, DATA_NUM):
        with open(input_path.format(str(i))) as f:
            for row in f:
                corr = re.match(pattern, row)
                if corr:
                    corr_list.append(float(corr.group(1)))
    print(predictor_name + ',' +
          str(round(mean(corr_list), 2))+',' +
          str(round(stdev(corr_list)/10, 5)))
    return corr_list


def main():
    df = pd.DataFrame()
    print('predictor name, mean, SE')
    for predictor in PREDICTOR_LIST:
        df[predictor] = collect_data_mono(predictor, INPUT_PATH)
    df['Cons3'] = collect_data_mono('cons3', INPUT_PATH)
    df['Cons4'] = collect_data_mono('cons4', INPUT_PATH)
    df['GPR'] = collect_data_mono('GPR', INPUT_PATH)
    df['RFR'] = collect_data_mono('RFR', INPUT_PATH)
    df['GPR_4'] = collect_data_mono('GPR_4', INPUT_PATH)
    df['RFR_4'] = collect_data_mono('RFR_4', INPUT_PATH)
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    df.to_csv(os.path.join(OUTPUT_DIR, 'all_CV.csv'), index=None)


if __name__ == '__main__':
    main()
