import pandas as pd
df=pd.read_csv("C:\\Users\\kimot\\OneDrive\\ドキュメント\\アプリ作成データ保管\\味の素(2802).csv",encoding='cp932')
import datetime as dt
df['Date']=pd.to_datetime(df['Date'])
df1=df[df['Date']<=dt.datetime(2024,1,17)]
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
# 特徴量とターゲット変数の選択
x = df1.drop(['株価','Date'],axis=1).values
t=df1['株価'].values
# 標準化
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
# データを訓練セットとテストセットに分割
x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.2, random_state=1)
# モデルの訓練
model = LinearRegression()
model.fit(x_train, t_train)
print(f'訓練データのaccuracy:{model.score(x_train,t_train)}')
print(f'テストデータのaccuracy:{model.score(x_test,t_test):.2f}')
import pickle
pickle.dump(model,open('model_stock','wb'))


