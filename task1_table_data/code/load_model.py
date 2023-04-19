"""課題１実行用ファイル"""

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier

# util
from utils import log_setting

logger = log_setting(__name__)

def load_machine_learning_model(select_model_name):
    """モデル読込処理"""

    logger.info('読込モデル %s', select_model_name)
    print(f'読込モデル {select_model_name}')

    if select_model_name == 'logreg' :
        # ロジスティック回帰
        return LogisticRegression()
    elif select_model_name == 'svc' :
        # SVM
        return SVC()
    elif select_model_name == 'linear_svc' :
        # 線形SVM
        return LinearSVC()
    elif select_model_name == 'decisiontree' :
        # 決定木
        return DecisionTreeClassifier()
    elif select_model_name == 'randomforest' :
        # ランダムフォレスト
        return RandomForestClassifier()
    elif select_model_name == 'knn' :
        # K-近傍法
        return KNeighborsClassifier()
    elif select_model_name == 'gaussian' :
        # ナイーブベイズ
        return GaussianNB()
    elif select_model_name == 'gbk' :      
        # 勾配ブースティング
        return GradientBoostingClassifier()
    elif select_model_name == 'sgd' :
        # 確率的勾配降下法
        return SGDClassifier()
    else:
        logger.error('正しくないモデル名が設定されています')
        print('正しくないモデル名が設定されています')
        return ''
