import os
import pandas as pd

def load_data(file_name):
    # 現在のファイルのディレクトリを取得
    current_dir = os.path.dirname(__file__)
    
    # CSVファイルへのパスを構築
    file_path = os.path.join(current_dir, 'data', file_name)

    # pandasを使用してCSVファイルを読み込む
    return pd.read_csv(file_path, encoding='cp932')
