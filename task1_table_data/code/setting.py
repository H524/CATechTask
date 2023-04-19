# 【設定パラメータ群】
# 交差検証を行うモデルの選択
SELECT_MODEL_ARR = ['decisiontree']
    # 'all', 'logreg', 'svc', 'linear_svc', 'decisiontree',
    # 'randomforest', 'knn', 'gaussian', 'gbk', 'sgd'

# 交差検証を行う際の分割方法の選択
CV_TYPE = 'kf'
    # 'all', 'kf', 'skf'

# 特徴量選択方法の選択
SELECT_NAME = ''
    # 'filter_method_low_dispersion', wrapper_method


# 【固定パラメータ群】
# モデル選択でallが選択された際に検証を行うモデル一覧
SELECT_MODEL_ALL = [
    'logreg', 'linear_svc', 'decisiontree',\
    'randomforest','knn', 'gaussian', 'gbk', 'sgd'
]
