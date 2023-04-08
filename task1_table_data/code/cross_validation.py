# import
import logging
from sklearn.model_selection import cross_val_score
import numpy as np

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s:%(name)s - %(message)s')
file_handler = logging.FileHandler('test.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def cross_validation(modelname, model, dataset):
    """交差検証を行う処理"""
    logging.info('start cross validation for %s', modelname)
    print('start cross validation for %s', modelname)
    scores = cross_val_score(model, dataset.data, dataset.target, cv=5)
#   np.savetxt(modelname + 'output_cv.csv', scores) 
#   np.savetxt(modelname + 'output_ave.csv', [np.mean(scores)])
    print(f'{modelname} Cross-Validation scores: {scores}')
    print(f'{modelname} Average score: {np.mean(scores)}')
    logging.info('end cross validation for %s', modelname)
    print('end cross validation for %s', modelname)