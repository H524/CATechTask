import sys
import logging
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier

# util
from cross_validation import cross_validation

arr = sys.argv[1:]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s:%(name)s - %(message)s')
file_handler = logging.FileHandler('test.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def task1(select_model_arr):
    """メイン処理"""
    # load dataset
    logging.info('start load dataset')
    print('start load dataset')
    dataset = datasets.fetch_covtype(data_home="./data")
    # x_train, x_test, t_train, t_test = train_test_split(dataset.data, dataset.target, test_size=0.3, random_state=0)

    logging.info('start cross validation')
    print('start cross validation')

    if select_model_arr in 'logreg':
        # ロジスティック回帰
        logreg = LogisticRegression()
        cross_validation('logreg', logreg, dataset)

    if select_model_arr in 'svc':
        # SVM
        svc = SVC()
        cross_validation('svc', svc, dataset)

    if select_model_arr in 'linear_svc':
        # LinearSVC
        linear_svc = LinearSVC()
        cross_validation('linear_svc', linear_svc, dataset)

    if select_model_arr in 'decisiontree':
        # 決定木
        decisiontree = DecisionTreeClassifier()
        cross_validation('decisiontree', decisiontree, dataset)

    if select_model_arr in 'randomforest':
        # ランダムフォレスト
        randomforest = RandomForestClassifier()
        cross_validation('randomforest', randomforest, dataset)
 
    if select_model_arr in 'knn':
        # K-近傍法
        knn = KNeighborsClassifier()
        cross_validation('knn', knn, dataset)

    if select_model_arr in 'gaussian':
        # ナイーブベイズ
        gaussian = GaussianNB()
        cross_validation('gaussian', gaussian, dataset)

    if select_model_arr in 'gbk':       
        # 勾配ブースティング
        gbk = GradientBoostingClassifier()
        cross_validation('gbk', gbk, dataset)

    if select_model_arr in 'logreg':
        # 確率的勾配降下法
        sgd = SGDClassifier()
        cross_validation('sgd', sgd, dataset)

    logging.info('end cross validation')
    print('end cross validation')









    # logging.info('end larning')
    # print('end larning')

    # logging.info('start fitting')
    # print('start fitting')

    # logging.info('fitting logreg')
    # print('fitting logreg')
    # # ロジスティック回帰
    # logreg.fit(x_train, t_train)
    # cross_validation('logreg_fit', logreg, dataset)
    # logreg_pred = logreg.predict(x_test) 
    # print('logreg_report_fit')
    # print(classification_report(t_test, logreg_pred))

    # logging.info('fitting decisiontree')
    # print('fitting decisiontree')
    # # 決定木
    # decisiontree.fit(x_train, t_train)
    # cross_validation('decisiontree_fit', decisiontree, dataset)
    # decisiontree_pred = decisiontree.predict(x_test)
    # print('logreg_fit_report')
    # print(classification_report(t_test, decisiontree_pred))

    # logging.info('fitting randomforest')
    # print('fitting randomforest')
    # # ランダムフォレスト
    # randomforest.fit(x_train, t_train)
    # cross_validation('randomforest_fit', randomforest, dataset)
    # randomforest_pred = randomforest.predict(x_test)
    # print('logreg_fit_report')
    # print(classification_report(t_test, randomforest_pred))

    # logging.info('end fitting')
    # print('end fitting')
    return

task1(arr)
