# About
This is a personal application of SHAP (SHapley Additive exPlanations: https://github.com/slundberg/shap) to analysis in relatively small memory machine environment like Google colab. This ongoing project tries to mitigate heavy memory-consumptions during SHAP calculations without affecting computational complexity.
# Outline of algorithm
0.Preparing input-datasets(Numpy array for features and a target)
1.Training models from full-records data(with cross-validation and averaging)
2.Splitting Full-records data into chunks of an empirically determined size(maximal records maintaining memory stability)
3.Predicting SHAP values and target values
4.Concatenating chunks and averaging over models(averaging is in step by step manner to reduce memory space)
5.Converting results into Pandas Dataframe, and appending column names and unique-keys(for convenience in following analysis)
