from fastapi import FastAPI
import pandas as pd
import pickle
from pydantic import BaseModel
import datetime as dt
from datetime import date as date_type
import os  # osモジュールをインポート
import numpy as np

app = FastAPI()

# モデルとデータセットのパスを定義
MODEL_PATH = 'model_stock'  # モデルファイルのパス
DATA_PATH = "C:\\Users\\kimot\\OneDrive\\ドキュメント\\アプリ作成データ保管\\味の素(2802).csv"  # データセットのパス

# 保存したモデルの読み込み
if os.path.exists(MODEL_PATH):
    model = pickle.load(open(MODEL_PATH, 'rb'))
else:
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

# データセットの読み込み
if os.path.exists(DATA_PATH):
    df2 = pd.read_csv(DATA_PATH, encoding='cp932')
else:
    raise FileNotFoundError(f"Data file not found at {DATA_PATH}")
df2['Date'] = pd.to_datetime(df2['Date'])
data = df2[df2['Date'] > dt.datetime(2024,1,17)]

# リクエストデータの型を定義
class PredictionRequest(BaseModel):
    Date: date_type
    

@app.post('/predict')
def predict_stock(request: PredictionRequest):
    selected_date = pd.to_datetime(request.Date)
    
    # 日付に対応する特徴量をデータセットから取得
    selected_data = data[data['Date'] == selected_date]
    selected_features = selected_data.drop(['Date', '株価'], axis=1)
    
    if not selected_features.empty:
        # モデルによる予測
        predicted_price = model.predict(selected_features)
        predicted_price = predicted_price[0] if predicted_price.size else None
        
        # 予測結果がNumPy型である場合、Pythonの標準型に変換
        if isinstance(predicted_price, np.integer):
            predicted_price = int(predicted_price)
        if isinstance(predicted_price, np.floating):
            predicted_price = int(predicted_price)

        # 特徴量の値を取得し、必要に応じてPython標準型に変換
        features = ['出来高', '売上高営業利益率', '営業利益成長率', '売上高成長率', '労働生産性成長率', '投下資本利益率', '研究開発比率', '為替レート']
        feature_values = {}
        for feature in features:
            value = selected_data[feature].values[0] if feature in selected_data else None
            if isinstance(value, np.integer):
                value = int(value)
            elif isinstance(value, np.floating):
                value = float(value)
            feature_values[feature] = value
        
        # 予測結果と特徴量の値を結果に含める
        result = {'予測株価': predicted_price}
        result.update(feature_values)
        
        return result  # この行は関数内にあることを確認
    else:
        return {"error": "選択された日付に対するデータが存在しません。"}  # この行も関数内にあることを確認

# この行は関数の外にあることを確認






