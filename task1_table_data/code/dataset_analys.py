import os
import logging
import pandas as pd
from sklearn import datasets
from sklearn.calibration import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s:%(name)s - %(message)s')
file_handler = logging.FileHandler('test.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

dataset = datasets.fetch_covtype(data_home="./data")

os.makedirs(f'output/dataset/analys', exist_ok=True)
data = pd.DataFrame(dataset.data, columns=dataset.feature_names)
output_data = pd.DataFrame({
        '最大値': data.max(),
        '最小値': data.min(),
        '平均値': data.mean()})
print(output_data)

# def output_result(modelname, cv_type, scores):
#     """交差検証の実行結果を出力する処理"""
#     os.makedirs(f'output/dataset/analys', exist_ok=True)
#     output_data = pd.DataFrame({
#         '学習時間（ｓ）': scores['fit_time'],
#         '評価時間（ｓ）': scores['score_time'],
#         '適合率': scores['test_f'],
#         '再現率': scores['test_p'],
#         'F値': scores['test_r'],
#         '正解率': scores['test_accuracy']})
#     output_data.loc['平均値']= output_data.mean()
#     output_data.round()
#     output_data.to_csv(f'output/{modelname}/{cv_type}/{modelname}_{cv_type}_output.csv')
#     print(output_data)

