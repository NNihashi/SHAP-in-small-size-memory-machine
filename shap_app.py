import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import shap

class SHAP_Calculation:

  def __init__(self, X, y, id_column=None, id_value=None):
    self.X = X
    self.y = y
    # Enter unique_key column if exists
    self.id_column = id_column
    self.id_value = id_value
    self.model_list = list()

  def training_model(self, cv_cnt, params, test_size=0.3, num_boost_round=10000, early_stopping_rounds=50):
    print("Model training")
    self.cv_cnt = cv_cnt
    # Creating models in Bootstrapping-cross-validation manner
    for loop_cnt in range(cv_cnt):
      # Splitting data into test/train randomly and repetitively
      X_train, X_test, y_train, y_test = train_test_split(
          self.X,
          self.y,
          test_size=test_size,
          random_state=loop_cnt
      )
      # test-data preparation
      X_train, X_valid, y_train, y_valid = train_test_split(
          X_train,
          y_train,
          test_size=test_size,
          random_state=loop_cnt
      )
      # training-data preparation
      d_train = lgb.Dataset(
          data=X_train,
          label=y_train
      )
      # valid-data preparation
      d_valid = lgb.Dataset(
          data=X_valid,
          label=y_valid
      )
      del X_train, y_train
      # training
      model =lgb.train(
        params=params,
        train_set=d_train,
        num_boost_round=num_boost_round,
        valid_sets=d_valid,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=1000
      )
      # validation by test_data
      y_pred_test = model.predict(X_test, pred_contrib=False)
      fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_test)
      auc = metrics.auc(fpr, tpr)
      print('test data auc: ', auc)
      del auc, fpr, tpr, thresholds, X_test, y_pred_test, d_train, d_valid
      # appending model into list
      self.model_list.append(model)
      del model

  def SHAP_Calculation(self, records_per_chunk=20000):
    print("SHAP calculation")
    total_records = len(self.X)
    print("total records: ", total_records)
    chunk_num = total_records // records_per_chunk + 1
    # SHAP will be calculated for each model.
    for model_index, model in enumerate(self.model_list):
      print("model index: ", model_index)
      # To save RAM space, SHAP calculation and target-prediction are to be done in limited numbers of records at once.
      for i in range(chunk_num):
        print("chunk: ", i)
        index_lower_lim = records_per_chunk * i
        index_upper_lim = records_per_chunk * (i+1)
        X_chunk = self.X[index_lower_lim:index_upper_lim]
        print("records per chunk: ", len(X_chunk))
        # SHAP computation(for each chunk)
        shap_values = model.predict(X_chunk, pred_contrib=True)
        # target prediction(for each chunk)
        # Reshaped for merge
        prediction = model.predict(X_chunk, pred_contrib=False).reshape((-1, 1))
        del X_chunk
        # shap and target prediction will be merged into one array(for each chunk)
        shap_with_prediction = np.concatenate([shap_values, prediction],
                                              axis=1)
        del shap_values, prediction
        if i == 0:
          shap_with_prediction_per_model = shap_with_prediction
        else:
          shap_with_prediction_per_model = np.concatenate([shap_with_prediction_per_model, shap_with_prediction],
                                                          axis=0)
      # Axis will be added for subsequent calculation(averaging SHAPs over all models)
      shap_with_prediction_per_model = shap_with_prediction_per_model.reshape((1,
                                                                               shap_with_prediction_per_model.shape[0],
                                                                               shap_with_prediction_per_model.shape[1]))
      # To optimize the memory space, averaging will be done in sequential manner.
      # Calculations will occupy only 2 data arrays of memory space, making the proceduces scalable.
      if model_index == 0:
        shap_with_prediction_all = shap_with_prediction_per_model
      else:
        shap_with_prediction_all = np.concatenate(
            [shap_with_prediction_all*model_index, shap_with_prediction_per_model],
            axis=0)
        shap_with_prediction_all = shap_with_prediction_all.mean(axis=0) * 2 / (model_index+1)
        # ndarray will be reshaped after averaging except for the last step.
        if model_index != self.cv_cnt - 1:
          shap_with_prediction_all = shap_with_prediction_all.reshape((1,
                                                                       shap_with_prediction_all.shape[0],
                                                                       shap_with_prediction_all.shape[1]))
          print('total record num: ', shap_with_prediction_all.shape[1])

    # Convertion of SHAP to dataframe
    shap_column_list = self.X.columns.values + "_shap"
    shap_column_list = shap_column_list.tolist()
    shap_column_list.append("base_value")
    shap_column_list.append("prediction")
    df_shap_lgb = (
        pd.DataFrame(
            shap_with_prediction_all,
            columns=shap_column_list
        )
    )
    # Actual target values and unique key will be implemented.
    df_shap_lgb["target"] = self.y
    if self.id_column != None:
      df_shap_lgb[self.id_column] = self.id_value

    return df_shap_lgb
