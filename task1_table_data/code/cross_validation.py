"""交差検証実行用ファイル"""

import os
import pandas as pd
from sklearn.calibration import cross_val_predict
from sklearn.feature_selection import SequentialFeatureSelector, VarianceThreshold
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier

from utils import log_setting

logger = log_setting(__name__)

def output_result(modelname, cv_type, scores):
    """交差検証の実行結果を出力する処理"""

    # 交差検証の結果格納先の作成
    os.makedirs(f'output/{modelname}/{cv_type}', exist_ok=True)

    # 交差検証の結果出力
    output_data = pd.DataFrame({
        '学習時間（ｓ）': scores['fit_time'],
        '評価時間（ｓ）': scores['score_time'],
        '適合率': scores['test_f'],
        '再現率': scores['test_p'],
        'F値': scores['test_r'],
        '正解率': scores['test_accuracy']})
    output_data.loc['平均値']= output_data.mean()
    output_data.round()
    output_data.to_csv(f'output/{modelname}/{cv_type}/{modelname}_{cv_type}_output.csv')

    logger.info('【出力完了】交差検証モデル： %s, 分割方法： %s', modelname, cv_type)
    print(f'【出力完了】交差検証モデル： {modelname}, 分割方法： {cv_type}')
    print(output_data)


def feature_select(dataset, select_name):
    """特徴量の選択を行う処理"""
    logger.info('特徴量の選択方法： %s', select_name)
    print(f'特徴量の選択方法： {select_name}')

    # 特徴量削減の出力結果格納先の作成
    os.makedirs(f'output/dataset/feature/filter_method/{select_name}', exist_ok=True)

    # 特徴量の削減を行う処理
    if select_name == 'filter_method_low_dispersion':
        selector = VarianceThreshold(threshold=(.8 * (1 - .8))) # ベルヌーイ分布の分散期待値を使った例
        dataset_new = selector.fit_transform(dataset.data)
    elif select_name == 'wrapper_method':
        dt = DecisionTreeClassifier()
        selector = SequentialFeatureSelector(dt, cv=5) #Backwardと共通関数で、Defaultがfoward
        dataset_new = selector.fit_transform(dataset.data, dataset.target)
    else :
        return dataset.data

    print('selected')
    result = pd.DataFrame(selector.get_support(), index=dataset.feature_names, columns=['False: dropped'])
    result.to_csv(f'output/dataset/feature/filter_method/{select_name}/output.csv')

    return dataset_new


def cross_validation(modelname, model, dataset, cv_type, select_name, cv_model):
    """交差検証を行う処理"""

    # 特徴量を選択する処理
    dataset_new = feature_select(dataset, select_name)

    logger.info('交差検証モデル： %s, 分割方法： %s', modelname, cv_type)
    print(f'交差検証モデル： {modelname}, 分割方法： {cv_type}')
    # 交差検証を行う処理
    scoring = {"p": "precision_macro",
                "r": "recall_macro",
                "f": "f1_macro",
                "accuracy": "accuracy",}
    scores = cross_validate(model, dataset_new, dataset.target, cv=cv_model, scoring=scoring, n_jobs=-1)
    output_result(modelname, cv_type, scores)

    # 交差検証の分類毎の精度を検証する処理
    pred = cross_val_predict(model, dataset_new, dataset.target, cv=cv_model, n_jobs=-1)
    print(classification_report(dataset.target, pred))

    logger.info('end cross validation for %s', modelname)
    print(f'end cross validation for {modelname}_{cv_type}')


def cross_validation_select(modelname, model, dataset, cv_type, select_name):
    """交差検証の分割方法を選択する処理"""
    logger.info('分割方法： %s', cv_type)
    print(f'分割方法： {cv_type}')

    if cv_type == 'kf' or cv_type == 'all':
        cv_model = KFold(n_splits=5, shuffle=True, random_state=0)
        cross_validation(modelname, model, dataset, cv_type, select_name, cv_model)

    if cv_type == 'skf' or cv_type == 'all':
        cv_model = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
        cross_validation(modelname, model, dataset, cv_type, select_name, cv_model)

    logger.info('end cross validation for %s', modelname)
    print(f'end cross validation for {modelname}_{cv_type}')
