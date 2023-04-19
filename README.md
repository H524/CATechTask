# CATechTask

# 課題１
## task_table_data/code 配下にコードを配置

## 実行手順は下記に記載（簡易表記）
1. covtypeデータセットの読み込みを実施する（ローカルに保存済みの場合にはローカルから読み込む）
1. 学習モデルの選択でallを選択していた場合、学習モデルを列挙した配列を生成する
1. 以降のフローは選択した学習モデルの数だけ実施 
1. 学習モデルの読込を実施
1. 交差検証の分割方法を選択（allを選択した場合には以降の処理をkfとskfでそれぞれ実施する）
1. 特徴量選択方法が指定されている場合、テストデータの特徴量削減を実施する
1. 交差検証を実施
1. 交差検証の実行結果をローカルファイルに出力

## ドキュメント系（提出資料等）は task_table_data/doc 配下に配置（実施はPrivateに変更後）