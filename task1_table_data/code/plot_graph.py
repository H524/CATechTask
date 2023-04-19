import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import seaborn as sns

dataset = datasets.fetch_covtype(data_home="./data")
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df["target"] = dataset.target

print(df["target"].count())
output_data = pd.DataFrame({
        'データ数': df.count(),
        '最大値': df.max(),
        '最小値': df.min(),
        '平均値': df.mean(),
        '欠損値（個数）': df.isnull().sum(),
        '=0': (df == 0).sum(),
        '=1': (df == 1).sum(),
        '=0,1': ((df == 0) | (df == 1)).sum(),
        'std': df.std(),
        })

data_count = [211840, 283301, 35754, 2747, 9493, 17367, 20510]
output_data_2 = pd.DataFrame({
        't1_0': (df.query('target == 1') == 0).sum()/data_count[0],
        't2_0': (df.query('target == 2') == 0).sum()/data_count[1],
        't3_0': (df.query('target == 3') == 0).sum()/data_count[2],
        't4_0': (df.query('target == 4') == 0).sum()/data_count[3],
        't5_0': (df.query('target == 5') == 0).sum()/data_count[4],
        't6_0': (df.query('target == 6') == 0).sum()/data_count[5],
        't7_0': (df.query('target == 7') == 0).sum()/data_count[6],
        })

# output_data_3 = pd.DataFrame({
#         't1_1': (df.query('target == 1') == 1).sum()/data_count[0],
#         't2_1': (df.query('target == 2') == 1).sum()/data_count[1],
#         't3_1': (df.query('target == 3') == 1).sum()/data_count[2],
#         't4_1': (df.query('target == 4') == 1).sum()/data_count[3],
#         't5_1': (df.query('target == 5') == 1).sum()/data_count[4],
#         't6_1': (df.query('target == 6') == 1).sum()/data_count[5],
#         't7_1': (df.query('target == 7') == 1).sum()/data_count[6]
#         })
# output_data_2 = pd.DataFrame({
#         'Soil_Type_0': (df.T["Soil_Type_0"] == 1).count(),
#         })
print(output_data.round(3))
print(output_data_2)
print(output_data_2.T.std())
# print(output_data_3.round(3))

sns.jointplot(x='Elevation', y='target', data=df, kind='hex')

# sns.pairplot(df, hue="target")
# print(dataset.keys())

os.makedirs(f'output/dataset/analys', exist_ok=True)

# print(dataset.DESCR)

# # # 結果を大きく描画する
# plt.rcParams['figure.figsize'] = (10.0, 10.0)

# # # 確認したい特徴量を記述
# # features = ['MedInc', 'AveOccup', 'HouseAge', 'AveRooms', # 目的変数との関係性を確認したい特徴量
# #             ('AveOccup', 'HouseAge'), # 特徴量間の関係性を可視化（下部のボックス）
# #             ('HouseAge', 'AveRooms') # 複数指定できる
# #            ]
# features = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area_0', 'Wilderness_Area_1', 'Wilderness_Area_2', 'Wilderness_Area_3', 'Soil_Type_0', 'Soil_Type_1', 'Soil_Type_2', 'Soil_Type_3', 'Soil_Type_4', 'Soil_Type_5', 'Soil_Type_6', 'Soil_Type_7', 'Soil_Type_8', 'Soil_Type_9', 'Soil_Type_10', 'Soil_Type_11', 'Soil_Type_12', 'Soil_Type_13', 'Soil_Type_14', 'Soil_Type_15', 'Soil_Type_16', 'Soil_Type_17', 'Soil_Type_18', 'Soil_Type_19', 'Soil_Type_20', 'Soil_Type_21', 'Soil_Type_22', 'Soil_Type_23', 'Soil_Type_24', 'Soil_Type_25', 'Soil_Type_26', 'Soil_Type_27', 'Soil_Type_28', 'Soil_Type_29', 'Soil_Type_30', 'Soil_Type_31', 'Soil_Type_32', 'Soil_Type_33', 'Soil_Type_34', 'Soil_Type_35', 'Soil_Type_36', 'Soil_Type_37', 'Soil_Type_38', 'Soil_Type_39']


# # # plot                        
# # plot_partial_dependence(estimator=randomforest, X=t_train, features=features,
# #                         n_jobs=-1, grid_resolution=20, n_cols=1)
# plot_partial_dependence(randomforest, t_train, features)



# # 特徴量の名前を取得する
features = dataset.feature_names

# # 目的変数の名前を取得する
target = dataset.target_names[0]

# # 各特徴量の分布をヒストグラムで可視化する
# for feature in features:
#     # print(len(dataset.data[:, dataset.feature_names.index(feature)]))
#     # print(len(dataset.target[dataset.target]))
#     # plt.scatter(dataset.data[:, dataset.feature_names.index(feature)], dataset.target)
#     # plt.pcolor(dataset.data[:, dataset.feature_names.index(feature)], dataset.target)
#     sns.jointplot(x=feature, y='target', data=df, kind='hex')
#     # plt.title(feature + ' vs ' + target)
#     # plt.xlabel(feature)
#     # plt.ylabel('target')
#     plt.savefig(f'output/dataset/analys/{feature}_histogram.png')  # グラフを保存する
# #     # plt.show()

