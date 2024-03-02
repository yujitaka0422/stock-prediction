import streamlit as st
import requests  # FastAPIにリクエストを送信するために必要
import pandas as pd
import datetime as dt
import numpy as np

st.title("株価予測アプリ")

# ユーザーに日付を入力してもらう
selected_date = st.date_input("予測する日付を選択してください")

# POSTリクエストのデータを準備
payload = {
    'Date': selected_date.isoformat()  # 日付をISO形式（YYYY-MM-DD）に変換
}

# FastAPIサーバーにPOSTリクエストを送信
url = 'http://127.0.0.1:8000/predict'  # URLの末尾に '/predict' を追加
# url = 'http://127.0.0.1:8000/predict'

try:
    response = requests.post(url, json=payload)
    # レスポンスのステータスコードを確認
    if response.status_code == 200:

        # レスポンスが成功した場合、データを抽出
        response_data = response.json()

        # レスポンスデータから各種数値を取得
        predicted_price = response_data.get('予測株価')
        features = {
            '出来高': response_data.get('出来高'),
            '売上高営業利益率': response_data.get('売上高営業利益率'),
            '営業利益成長率': response_data.get('営業利益成長率'),
            '売上高成長率': response_data.get('売上高成長率'),
            '労働生産性成長率': response_data.get('労働生産性成長率'),
            '投下資本利益率': response_data.get('投下資本利益率'),
            '研究開発比率': response_data.get('研究開発比率'),
            '為替レート': response_data.get('為替レート')
        }

        # 予測された株価を表示
        if predicted_price is not None:
            st.metric(label="予測株価", value=f"{predicted_price}")
        else:
            st.error("株式相場はお休みです。")
        
        # 特徴量を表形式で表示
        features_df = pd.DataFrame(features.items(), columns=['特徴量', '値'])
        st.table(features_df)

    else:
        # レスポンスが成功しなかった場合、エラーメッセージを表示
        error_message = response.json().get('detail', '株式相場はお休みです。')
        st.error(error_message)
except requests.exceptions.ConnectionError:
        st.error("FastAPIサーバーに接続できません。サーバーが起動していることを確認してください。")





    
