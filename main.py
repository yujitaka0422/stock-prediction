from fastapi import FastAPI, HTTPException
import pandas as pd
import pickle
import numpy as np
from pydantic import BaseModel
import datetime as dt
from data_loader import load_data  # data_loader.py から関数をインポート
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # すべてのドメインからのアクセスを許可
    allow_credentials=True,
    allow_methods=["*"],  # すべてのHTTPメソッドを許可
    allow_headers=["*"],  # すべてのHTTPヘッダーを許可
)

MODEL_PATH = 'model_stock'

if os.path.exists(MODEL_PATH):
    model = pickle.load(open(MODEL_PATH, 'rb'))
else:
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

# CSVファイルの読み込み
try:
    df2 = load_data('味の素(2802).csv')  # CSVファイル名を指定して関数を呼び出し
except FileNotFoundError as e:
    raise FileNotFoundError(f"Data file not found: {e}")

df2['Date'] = pd.to_datetime(df2['Date'])
data = df2[df2['Date'] > dt.datetime(2024,1,17)]

class PredictionRequest(BaseModel):
    Date: dt.date

@app.post('/predict')
async def predict_stock(request: PredictionRequest):
    selected_date = pd.to_datetime(request.Date)
    selected_data = data[data['Date'] == selected_date]

    if selected_data.empty:
        raise HTTPException(status_code=404, detail="Data for the selected date is not available.")

    selected_features = selected_data.drop(['Date', '株価'], axis=1)

    predicted_price = model.predict(selected_features)
    predicted_price = predicted_price[0] if predicted_price.size else None

    # Check if predicted_price is a numpy type, if yes, convert to Python type
    if isinstance(predicted_price, (np.integer, np.floating)):
        predicted_price = int(predicted_price.item())

    features = ['出来高', '売上高営業利益率', '営業利益成長率', '売上高成長率', '労働生産性成長率', '投下資本利益率', '研究開発比率', '為替レート']
    feature_values = {}
    for feature in features:
        value = selected_data[feature].values[0] if feature in selected_data else None
        # Check if value is a numpy type, if yes, convert to Python type
        if isinstance(value, (np.integer, np.floating)):
            value = value.item()
        feature_values[feature] = value

    result = {'予測株価': predicted_price}
    result.update(feature_values)

    return result



