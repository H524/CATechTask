# logging
import logging

# import
# dataset
from sklearn import datasets
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier

# util
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import precision_score
# from sklearn.metrics import recall_score
# from sklearn.metrics import classification_report
from cross_validation import cross_validation


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s:%(name)s - %(message)s')
file_handler = logging.FileHandler('test.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def task1():
    """メイン処理"""
    # load dataset
    logging.info('start load dataset')
    print('start load dataset')
    dataset = datasets.fetch_covtype()
    # x_train, x_test, t_train, t_test = train_test_split(dataset.data, dataset.target, test_size=0.3, random_state=0)

    # load model
    logging.info('start load model')
    print('tart load model')
    logreg = LogisticRegression()
    svc = SVC()
    linear_svc = LinearSVC()
    decisiontree = DecisionTreeClassifier()
    randomforest = RandomForestClassifier()
    knn = KNeighborsClassifier()
    gaussian = GaussianNB()
    gbk = GradientBoostingClassifier()
    sgd = SGDClassifier()

    logging.info('start larning')
    print('start larning')
    # ロジスティック回帰
    cross_validation('logreg', logreg, dataset)
    # SVM
    cross_validation('svc', svc, dataset)
    # LinearSVC
    cross_validation('linear_svc', linear_svc, dataset)
    # 決定木
    cross_validation('decisiontree', decisiontree, dataset)
    # ランダムフォレスト
    cross_validation('randomforest', randomforest, dataset)
    # KNeighborsClassifier
    cross_validation('knn', knn, dataset)
    # ナイーブベイズ
    cross_validation('gaussian', gaussian, dataset)
    # GradientBoostingClassifier
    cross_validation('gbk', gbk, dataset)
    # Stochastic Gradient Descent
    cross_validation('sgd', sgd, dataset)
    logging.info('end larning')
    print('end larning')
    return

task1()
