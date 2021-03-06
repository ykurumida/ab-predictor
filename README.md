#         Predicting Antibody Affinity Changes upon Mutations by Combining Multiple Predictors  







## Required Packages

The following is the list of required packeages and programs, as well as the version on which it was tested (in parenthesis).
- Python (3.7.2)
- numpy (1.16.3)
- pandas (0.24.2)
- scikit-learn (0.21.2)
- statistics (1.0.3.5)



## SiPMAB

```bash
python corr_run.py ../Input/ddG.csv [SEED] # calculate peason's correlation each seed
python analysis.py # calculate mean and SE of the peason's correlation
python fi_analysis.py # calculate feature importance
```

## Independent data

 ```bash
 python pred_holdout.py ../Input/test_2bdn_3bdy.csv ${SEED}
 ```


## Reference

Kurumida Y, Saito Y, Kameda T. Predicting antibody affinity changes upon mutations by combining multiple predictors. Sci Rep. 2020 Nov 11;10(1):19533. doi: 10.1038/s41598-020-76369-8. 

