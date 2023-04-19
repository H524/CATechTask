"""共通処理用ファイル"""

import logging

def log_setting(level):
    """ログ出力の初期設定を行う処理"""
    # logging
    logger = logging.getLogger(level)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s:%(name)s - %(message)s')
    file_handler = logging.FileHandler('test.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger