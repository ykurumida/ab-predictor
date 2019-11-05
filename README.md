#         AB-MBLAP (AntiBody Machine Learning Based Affinity Prediction)  







##Required Packages

- Python 2.7.5 or 3.7.2
- numpy >= 1.16.3
- pandas >= 0.24.2
- scikit-learn >= 0.21.2
- statistics >= 1.0.3.5



## Usage

```bash
python corr_run.py ../Input/ddG.csv [SEED] # calculate peason's correlation each seed
python analysis.py # calculate mean and SE of the peason's correlation
python fi_analysis.py # calculate feature importance
```



## Reference

Y. Kurumida, Y. Saito, and T. Kameda (Submitted)
