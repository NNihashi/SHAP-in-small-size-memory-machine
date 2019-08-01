# About
This is a trivial application of SHAP (SHapley Additive exPlanations: https://github.com/slundberg/shap) to analysis under relatively small memory machine environment like Google colab. This ongoing project tries to mitigate severe memory-consumptions during SHAP calculations of large scale data without affecting computational complexity.  
Because this is a personal project, contents are to be modified Without notice.
# Outline of algorithm
0. Preparing input-datasets(Numpy array for features(X) and a target(y))  
1. Training models(LightGBM: https://github.com/microsoft/LightGBM) from full-records data(with cross-validation and averaging for enhancing statistical significance and generalization performance)  
2. Splitting Full-records data into chunks of an empirically determined size(maximal records maintaining memory stability)  
3. Predicting SHAP values and target values  
4. Concatenating chunks and averaging over models(averaging is in step by step manner to spare memory space)  
5. Converting results into Pandas Dataframe, and appending column names and unique-keys(for convenience in following analysis)
# How to use(Example: binary classification)  
`from shap_app import SHAP_Calculation`  
`params = {
    "max_bin": 512,
    "learning_rate": 0.05,
    "boosting_type": "gbdt",
    "objective": "binary",
    "num_leaves": 10,
    "verbose": -1,
    "min_data": 10,
    "boost_from_average": True,
    "metric": 'auc'
}`  
`cv_cnt = 5`  
`SHAP_lgbm = SHAP_Calculation(X, y)`  
`SHAP_lgbm.training_model(cv_cnt, params)`  
`shap = SHAP_lgbm.SHAP_Calculation()`  



