"""課題１実行用ファイル"""

from sklearn import datasets
from setting import CV_TYPE, SELECT_MODEL_ALL, SELECT_MODEL_ARR, SELECT_NAME

from load_model import load_machine_learning_model
from cross_validation import cross_validation_select
from utils import log_setting

logger = log_setting(__name__)

def model_select(select_model_arr, modelname):
    """モデルの選択判定処理"""
    return modelname in select_model_arr or 'all' in select_model_arr

def task1(select_model_arr, cv_type, select_name):
    """メイン処理"""
    # データセットの読込
    logger.info('データセットの読込開始')
    print('データセットの読込開始')
    dataset = datasets.fetch_covtype(data_home="./data")

    # モデル選択でallが選択された場合の処理
    if 'all' in select_model_arr:
        select_model_arr = SELECT_MODEL_ALL

    # 機械学習の実施
    for select_model_name in select_model_arr:
        # 機械学習モデルの読込
        logger.info('モデル読込開始')
        print('モデル読込開始')
        model = load_machine_learning_model(select_model_name)

        if model != '':
            # 交差検証の実施
            logger.info('交差検証開始')
            print('交差検証開始')
            cross_validation_select(select_model_name, model, dataset, cv_type, select_name)
        else:
            logger.error('正しくないモデル名が設定されています')
            print('正しくないモデル名が設定されています')

    logger.info('完了')
    print('完了')
    return

task1(SELECT_MODEL_ARR, CV_TYPE, SELECT_NAME)
