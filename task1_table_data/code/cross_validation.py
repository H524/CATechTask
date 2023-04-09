import os
import time
import logging
from pprint import pprint
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_validate

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s:%(name)s - %(message)s')
file_handler = logging.FileHandler('test.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def cross_validation(modelname, model, dataset):
    """交差検証を行う処理"""
    start_time = time.perf_counter()
    logging.info('start cross validation for %s', modelname)
    print(f'start cross validation for {modelname}')

    # scores = cross_val_score(model, dataset.data, dataset.target, cv=5)
    scoring = {"p": "precision_macro",
                "r": "recall_macro",
                "f": "f1_macro"}
    skf = StratifiedKFold(n_splits=5, shuffle=False)
    scores = cross_validate(model, dataset.data, dataset.target, cv=skf, scoring=scoring)

    pprint(scores)

    os.makedirs(f'output/{modelname}', exist_ok=True)
    # np.savetxt(f'output/{modelname}_cv.csv', np.round(scores, decimals=3))
    np.savetxt(f'output/{modelname}/{modelname}_fit_time.csv', np.round(scores['fit_time'], decimals=3))
    np.savetxt(f'output/{modelname}/{modelname}_score_time.csv', np.round(scores['score_time'], decimals=3))
    np.savetxt(f'output/{modelname}/{modelname}_test_f.csv', np.round(scores['test_f'], decimals=3))
    np.savetxt(f'output/{modelname}/{modelname}_test_p.csv', np.round(scores['test_p'], decimals=3))
    np.savetxt(f'output/{modelname}/{modelname}_test_r.csv', np.round(scores['test_r'], decimals=3))
    # np.savetxt(f'output/{modelname}_ave.csv', [round(np.mean(scores), 3)])

    # print(f'{modelname} Cross-Validation scores: {np.round(scores, decimals=3)}')
    # print(f'{modelname} Average score: {np.mean(scores):.3f}')
    print(f'Execution time: {time.perf_counter() - start_time:.1f}(s)')

    logging.info('end cross validation for %s', modelname)
    print(f'end cross validation for {modelname}')
