# ライブラリのインポート
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os,sys

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

from lightgbm import LGBMClassifier
import pickle as pkl

from lime.lime_tabular import LimeTabularExplainer

# データの読み込み
train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')


# NameをTitleに変換する関数
def replace_name(series):
    series = series.apply(lambda x: x.split(',')[1].split('.')[0].strip())
    series = series.replace(['Capt','Col','Major','Dr','Rev'], 'Officer')
    series = series.replace(['Don','Sir','the Countess','Lady','Dona'], 'Royalty')
    series = series.replace(['Mme','Ms'], 'Mrs')
    series = series.replace(['Mlle'], 'Miss')
    series = series.replace(['Jonkheer'], 'Master')
    return series


# カテゴリカル変数の定義
categorical_features = ["Sex", "Embarked", "Title"]
# 特徴量エンジニアリングの実行
train = train.dropna(subset=['Embarked'])
df = pd.concat([train, test])
df['Family'] = df['SibSp'] + df['Parch'] + 1
df["Title"] = replace_name(df["Name"])
df["Age_null"] = df["Age"].isnull()
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Fare"] = df["Fare"].fillna(df[df["Pclass"]==3]["Fare"].median())
df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch'])
oe = OrdinalEncoder()
df[categorical_features] = oe.fit_transform(df[categorical_features]).astype(int)
train = df[df["Survived"].notnull()]
test = df[df["Survived"].isnull()].drop(columns=["Survived"])
# 訓練データを学習データと検証データに分割
train, valid = train_test_split(train, test_size=0.2, stratify=train["Survived"], random_state=100)
# データの保存
train.to_csv("dataset/train_proc.csv",index=None)
valid.to_csv("dataset/valid_proc.csv",index=None)
test.to_csv("dataset/test_proc.csv",index=None)


# 説明変数と目的変数に分割
def make_Xy(df, col_y="Survived"):
    return df.drop(columns=[col_y]), df[col_y]


train_X, train_y = make_Xy(train)
valid_X, valid_y = make_Xy(valid)
test_X = test

# モデルの定義
model = LGBMClassifier(max_depth=4, colsample_bytree=0.5,
                       reg_lambda=0.5, reg_alpha=0.5,
                       importance_type="gain", random_state=100)

# 学習の実行
model.fit(
    train_X, train_y,
    eval_set=[(valid_X, valid_y)],
    categorical_feature=categorical_features
)

# 正解率の計算
print("Accuracy(train) {:.3f}".format(model.score(train_X, train_y)))
print("Accuracy(valid) {:.3f}".format(model.score(valid_X, valid_y)))

# モデルの保存
with open('model/lgbm_model.pkl', 'wb') as f:
    pkl.dump(model, f)


# LIME用の予測関数の準備
def predict_fn(X):
    if len(X.shape)==1:
        return model.predict_proba(X.reshape(1,-1))[0]
    else:
        return model.predict_proba(X)


# 数値とカテゴリの対応の準備
class_names = ["Not Survived", "Survived"]
categorical_feature_idx = np.where(train_X.columns.isin(categorical_features))[0]
categorical_names = dict(zip(categorical_feature_idx , [list(lst) for lst in oe.categories_]))
print(categorical_feature_idx)
print(categorical_names)

# 説明用クラスの定義
explainer = LimeTabularExplainer(training_data=train_X.values,
                                 feature_names=train_X.columns,
                                 categorical_features=categorical_feature_idx,
                                 categorical_names=categorical_names,
                                 class_names=class_names
                                )

# 予測対象データのインデックス
i = 0

# 局所説明の計算
exp = explainer.explain_instance(test_X.values[i], predict_fn, num_features=5)
# matplotlib形式での取得
plt = exp.as_pyplot_figure()
plt.savefig('result/titanic_lime_matplotlib.png', bbox_inches='tight')

# 分析用テーブルを作成
pred = predict_fn(test_X)
result = pd.DataFrame({"pred":pred[:,0], "Sex":test["Sex"]})
# 男性なのにSurviveと予測する例を取得
target_idx = result[result["Sex"]==1].sort_values("pred").index
# LIMEの実行
i = target_idx[0]
exp = explainer.explain_instance(test_X.values[i], predict_fn, num_features=5)
# matplotlib形式での取得
plt = exp.as_pyplot_figure()
plt.savefig('result/titanic_lime2_matplotlib.png', bbox_inches='tight')
# html形式での取得
result_html = exp.as_html()
with open("result/result_html.html", "w", encoding='utf-8') as f:
    f.write(result_html)