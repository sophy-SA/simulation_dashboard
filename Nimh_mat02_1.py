import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import sys

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# 認証用の関数
def verify_credentials(username, password):
    valid_credentials = {
        "kai01": "pass_kai",
        "cae01": "pass_cae"
    }
    return username in valid_credentials and valid_credentials[username] == password

def check_authentication():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        
    if not st.session_state.authenticated:
        username = st.text_input("ユーザー名")
        password = st.text_input("パスワード", type="password")
        
        if st.button("ログイン"):
            if verify_credentials(username, password):
                st.session_state.authenticated = True
                return True
            else:
                st.error("認証に失敗しました")
                return False
        return False
    
    return True

# メイン処理の前に、必要な関数を定義
def show_model_management():
    st.header("学習モデルの作成と管理")
    
    # モデルリストの表示
    st.subheader("保存済みモデル一覧")
    model_files = os.listdir("mat_model")
    if model_files:
        for model in model_files:
            st.write(f"- {model}")
    else:
        st.write("保存されているモデルはありません")
    
    # モデル作成セクション
    st.subheader("新規モデルの作成")
    data_files = os.listdir("mat_data")
    selected_file = st.selectbox("学習データの選択", data_files)
    
    model_name = st.text_input("モデル名", "model_name")
    
    if st.button("学習開始"):
        with st.spinner("学習中..."):
            try:
                trainer = AlloyModelTrainer()
                models, scalers = trainer.train_models(f"mat_data/{selected_file}")
                
                # モデルの保存
                save_path = f"mat_model/{model_name}"
                os.makedirs(save_path, exist_ok=True)
                
                # モデルと前処理用スケーラーを保存
                for target, model in models.items():
                    torch.save(model.state_dict(), f"{save_path}/{target}_model.pth")
                joblib.dump(scalers, f"{save_path}/scalers.pkl")
                
                st.success("モデルの学習と保存が完了しました")
            except Exception as e:
                st.error(f"エラーが発生しました: {str(e)}")
    
    # モデル削除セクション
    st.subheader("モデルの削除")
    model_to_delete = st.selectbox("削除するモデルの選択", model_files if model_files else ["モデルがありません"])
    
    if st.button("モデル削除") and model_files:
        try:
            import shutil
            shutil.rmtree(f"mat_model/{model_to_delete}")
            st.success(f"モデル {model_to_delete} を削除しました")
            st.rerun()
        except Exception as e:
            st.error(f"削除中にエラーが発生しました: {str(e)}")

def show_prediction():
    st.header("特性予測の実行")
    
    # モデル選択
    model_files = os.listdir("mat_model")
    if not model_files:
        st.error("利用可能なモデルがありません。先にモデルを作成してください。")
        return
    
    selected_model = st.selectbox("予測に使用するモデル", model_files)
    
    # 元素リストの定義
    elements = ['La', 'Ce', 'Pr', 'Nd', 'Sm', 'Y', 'Ca', 'Zr', 
               'Mg', 'Ni', 'Co', 'Al', 'Mn', 'Zn', 'Cr']
    re_elements = ['La', 'Ce', 'Pr', 'Nd', 'Sm', 'Y', 'Zr', 'Mg']  # 合計1.0にする元素群
    
    # スイープする元素の選択
    sweep_elements = st.multiselect("スイープする元素を選択（最大2つ）", 
                                  elements, 
                                  max_selections=2)
    
    # 調整元素の選択
    adjust_element = st.selectbox("組成調整する元素を選択", 
                                [e for e in re_elements if e not in sweep_elements],
                                key="adjust_element")
    
    # 警告表示の条件チェック
    warning = False
    if adjust_element in sweep_elements:
        st.warning("調整元素はスイープ元素と重複して選択できません")
        warning = True
    
    # 入力値の設定
    input_values = {}
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("固定値の設定")
        for elem in elements:
            if elem not in sweep_elements:
                input_values[elem] = st.number_input(f"{elem}の値",
                                                   value=0.0, 
                                                   format="%.2f")
    
    with col2:
        st.subheader("スイープ範囲の設定")
        sweep_ranges = {}
        for elem in sweep_elements:
            st.write(f"{elem}のスイープ範囲")
            start = st.number_input(f"{elem} 開始値", 
                                  value=0.0, 
                                  format="%.2f", 
                                  key=f"{elem}_start")
            end = st.number_input(f"{elem} 終了値", 
                                value=1.0, 
                                format="%.2f", 
                                key=f"{elem}_end")
            sweep_ranges[elem] = (start, end)
    
    # 警告表示の条件チェック
    warning = False
    warning_messages = []
    
    if adjust_element in sweep_elements:
        warning_messages.append("調整元素はスイープ元素と重複して選択できません")
        warning = True
    
    # 組成制限のチェック
    composition_limits = {
        'Ca': 0.001,
        'Co': 0.0,
        'Al': 0.4,
        'Mn': 0.5,
        'Zn': 0.3,
        'Cr': 0.1
    }
    
    if len(sweep_elements) == 0:
        # スイープなしの場合、固定値で直接チェック
        for elem, limit in composition_limits.items():
            if input_values.get(elem, 0) > limit:
                warning_messages.append(f"{elem}の値が{limit}より大きいため実行できません")
                warning = True
    else:
        # スイープがある場合、固定値とスイープ範囲の最大値をチェック
        for elem, limit in composition_limits.items():
            if elem in sweep_elements:
                # スイープ対象の場合、範囲の最大値をチェック
                _, end = sweep_ranges[elem]
                if end > limit:
                    warning_messages.append(f"{elem}のスイープ範囲の最大値が{limit}より大きいため実行できません")
                    warning = True
            elif input_values.get(elem, 0) > limit:
                # 固定値の場合
                warning_messages.append(f"{elem}の値が{limit}より大きいため実行できません")
                warning = True
    
    # 警告メッセージの表示
    for message in warning_messages:
        st.warning(message)
    
    if st.button("予測実行") and not warning:
        try:
            # スケーラーの読み込み
            scalers = joblib.load(f"mat_model/{selected_model}/scalers.pkl")
            
            # 合計1超過フラグ
            total_exceeded = False
            
            # スイープ設定に基づいてデータ生成
            if len(sweep_elements) == 0:
                # スイープなしの場合
                # RE元素の合計を計算（調整元素を除く）
                re_sum = sum(input_values[elem] for elem in re_elements if elem != adjust_element)
                # 調整元素の値を設定
                input_values[adjust_element] = 1.0 - re_sum
                
                if input_values[adjust_element] < 0:
                    total_exceeded = True
                    results = {key: 0 for key in ['Temp', 'cap', 'iziritsu', 'density', 'Pd80_04', 'Pd80_05']}
                else:
                    input_data = np.array([[input_values[elem] for elem in elements]])
                    results = predict_properties(f"mat_model/{selected_model}", 
                                              input_data, 
                                              scalers)
                
                # 結果をDataFrameに変換
                results_df = pd.DataFrame([results])
                
            elif len(sweep_elements) == 1:
                # 1次元スイープの場合
                elem = sweep_elements[0]
                start, end = sweep_ranges[elem]
                sweep_values = np.linspace(start, end, 100)
                
                results_list = []
                for val in sweep_values:
                    input_values[elem] = val
                    # RE元素の合計を計算（調整元素を除く）
                    re_sum = sum(input_values[e] for e in re_elements if e != adjust_element)
                    # 調整元素の値を設定
                    input_values[adjust_element] = 1.0 - re_sum
                    
                    if input_values[adjust_element] < 0:
                        total_exceeded = True
                        results = {key: 0 for key in ['Temp', 'cap', 'iziritsu', 'density', 'Pd80_04', 'Pd80_05']}
                    else:
                        input_data = np.array([[input_values[e] for e in elements]])
                        results = predict_properties(f"mat_model/{selected_model}", 
                                                  input_data, 
                                                  scalers)
                    results[elem] = val
                    results_list.append(results)
                
                results_df = pd.DataFrame(results_list)
                
            else:
                # 2次元スイープの場合
                elem1, elem2 = sweep_elements
                start1, end1 = sweep_ranges[elem1]
                start2, end2 = sweep_ranges[elem2]
                
                sweep_values1 = np.linspace(start1, end1, 50)
                sweep_values2 = np.linspace(start2, end2, 50)
                
                results_list = []
                for val1 in sweep_values1:
                    for val2 in sweep_values2:
                        input_values[elem1] = val1
                        input_values[elem2] = val2
                        # RE元素の合計を計算（調整元素を除く）
                        re_sum = sum(input_values[e] for e in re_elements if e != adjust_element)
                        # 調整元素の値を設定
                        input_values[adjust_element] = 1.0 - re_sum
                        
                        if input_values[adjust_element] < 0:
                            total_exceeded = True
                            results = {key: 0 for key in ['Temp', 'cap', 'iziritsu', 'density', 'Pd80_04', 'Pd80_05']}
                        else:
                            input_data = np.array([[input_values[e] for e in elements]])
                            results = predict_properties(f"mat_model/{selected_model}", 
                                                      input_data, 
                                                      scalers)
                        results[elem1] = val1
                        results[elem2] = val2
                        results_list.append(results)
                
                results_df = pd.DataFrame(results_list)
            
            # 結果の保存
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            if len(sweep_elements) == 1:
                filename = f"predict/prediction_sweep1d_{sweep_elements[0]}_{timestamp}.csv"
            elif len(sweep_elements) == 2:
                filename = f"predict/prediction_sweep2d_{sweep_elements[0]}_{sweep_elements[1]}_{timestamp}.csv"
            else:
                filename = f"predict/prediction_{timestamp}.csv"
            results_df.to_csv(filename, index=False)
            st.success(f"予測結果を保存しました: {filename}")
            
            if total_exceeded:
                st.warning("A原子合計が1を超えた条件があります")
                
        except Exception as e:
            st.error(f"予測中にエラーが発生しました: {str(e)}")
            



def show_analysis():
    st.header("結果の表示と分析")
    
    # 予測結果ファイルの一覧表示
    predict_files = os.listdir("predict")
    if not predict_files:
        st.error("予測結果ファイルが存在しません")
        return
    
    selected_file = st.selectbox("分析する結果ファイルを選択", predict_files)
    
    try:
        df = pd.read_csv(f"predict/{selected_file}")
        
        # ファイルタイプの判定（ファイル名から判定）
        if "sweep1d" in selected_file:
            show_1d_analysis(df)
        elif "sweep2d" in selected_file:
            show_2d_analysis(df)
        else:
            show_single_prediction(df)
            
    except Exception as e:
        st.error(f"結果の表示中にエラーが発生しました: {str(e)}")

    # ファイル削除機能の追加
    st.markdown("---")  # 区切り線
    st.subheader("予測結果ファイルの削除")
    file_to_delete = st.selectbox("削除するファイルを選択", predict_files, key="delete_file")
    
    if st.button("ファイル削除"):
        try:
            os.remove(f"predict/{file_to_delete}")
            st.success(f"ファイル {file_to_delete} を削除しました")
            st.rerun()  # ページを更新して変更を反映
        except Exception as e:
            st.error(f"削除中にエラーが発生しました: {str(e)}")

def show_single_prediction(df):
    st.subheader("予測結果")
    
    # データを見やすく整形して表示
    results_table = pd.DataFrame()
    for col in df.columns:
        results_table[col] = [f"{df[col].iloc[0]:.6f}"]
    
    st.write(results_table.T)

def show_1d_analysis(df):
    st.subheader("1次元スイープ分析")

    # スイープした元素を特定（最終列を使用）
    sweep_element = df.columns[-1]
    
    # プロット対象の特性値を選択（1つ目のグラフ）
    plot_properties = [col for col in df.columns if col != sweep_element]
    selected_property = st.selectbox("表示する特性値を選択", plot_properties)
    
    # プロットの作成（1つ目のグラフ）
    fig = px.line(df, x=sweep_element, y=selected_property,
                  title=f"{selected_property} vs {sweep_element}")
    
    # グラフのレイアウト設定を更新
    fig.update_layout(
        margin=dict(l=20, r=20),
        title=dict(
            text=f"{selected_property} vs {sweep_element}",
            font=dict(size=26, weight='bold')  # タイトルフォントサイズを26ptに
        ),
        xaxis=dict(
            title=dict(
                text=f"{sweep_element} 組成",
                font=dict(size=20, weight='bold')  # X軸ラベルフォントサイズを20ptに
            ),
            tickfont=dict(size=16),  # X軸目盛りフォントサイズを16ptに
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            showline=True,
            linewidth=2,
            linecolor='black'
        ),
        yaxis=dict(
            title=dict(
                text=f"{selected_property} 値",
                font=dict(size=20, weight='bold')  # Y軸ラベルフォントサイズを20ptに
            ),
            tickfont=dict(size=16),  # Y軸目盛りフォントサイズを16ptに
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            showline=True,
            linewidth=2,
            linecolor='black'
        ),
        plot_bgcolor='white',
        showlegend=False,
        width=800,
        height=600
    )
    
    # プロットエリアの枠線を追加
    fig.update_xaxes(mirror=True)
    fig.update_yaxes(mirror=True)
    
    st.plotly_chart(fig)
    
    # 数値データの表示
    if st.checkbox("数値データを表示"):
        st.write(df[[sweep_element, selected_property]])
    
    
    # 2つ目のグラフの設定
    st.subheader("カスタムプロット")
    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("X軸を選択", df.columns)
    with col2:
        y_axis = st.selectbox("Y軸を選択", df.columns)
    
    # プロットスタイルの選択
    plot_style = st.selectbox("プロットスタイルを選択", ["ライン", "ドット"])
    
    # プロットの作成
    if plot_style == "ライン":
        fig2 = px.line(df, x=x_axis, y=y_axis)
    else:
        fig2 = px.scatter(df, x=x_axis, y=y_axis)
    
    fig2.update_layout(
        margin=dict(l=20, r=20),
        title=dict(
            text=f"{y_axis} vs {x_axis}",
            font=dict(size=26, weight='bold')  # タイトルフォントサイズを26ptに
        ),
        xaxis=dict(
            title=dict(
                text=f"{x_axis}",
                font=dict(size=20, weight='bold')  # X軸ラベルフォントサイズを20ptに
            ),
            tickfont=dict(size=16),  # X軸目盛りフォントサイズを16ptに
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            showline=True,
            linewidth=2,
            linecolor='black'
        ),
        yaxis=dict(
            title=dict(
                text=f"{y_axis}",
                font=dict(size=20, weight='bold')  # Y軸ラベルフォントサイズを20ptに
            ),
            tickfont=dict(size=16),  # Y軸目盛りフォントサイズを16ptに
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            showline=True,
            linewidth=2,
            linecolor='black'
        ),
        plot_bgcolor='white',
        showlegend=False,
        width=800,
        height=600
    )
    
    # プロットエリアの枠線を追加
    fig2.update_xaxes(mirror=True)
    fig2.update_yaxes(mirror=True)
    
    st.plotly_chart(fig2)
    


def show_2d_analysis(df):
    st.subheader("2次元スイープ分析")
    
    # スイープした2つの元素を特定（最後の2列を使用）
    sweep_elements = list(df.columns[-2:])
    
    # プロット対象の特性値を選択
    plot_properties = [col for col in df.columns if col not in sweep_elements]
    selected_property = st.selectbox("表示する特性値を選択", plot_properties)
    
    # カラーマップの範囲設定
    col1, col2 = st.columns(2)
    with col1:
        # 0を除いた最小値を設定
        min_value = df[selected_property][df[selected_property] > 0].min()
        vmin = st.number_input("最小値（青）", 
                              value=float(min_value),
                              format="%.2f")  # 小数点2桁に設定
    with col2:
        vmax = st.number_input("最大値（赤）", 
                              value=float(df[selected_property].max()),
                              format="%.2f")  # 小数点2桁に設定
    
    # 2Dマップの作成
    x_unique = sorted(df[sweep_elements[0]].unique())
    y_unique = sorted(df[sweep_elements[1]].unique())
    z_matrix = df[selected_property].values.reshape(len(y_unique), len(x_unique))
    
    # 0の値を除外してコンターを作成
    z_matrix[z_matrix == 0] = np.nan  # 0の値をNaNに置き換え
    
    fig = go.Figure(data=go.Heatmap(
        z=z_matrix,
        x=x_unique,
        y=y_unique,
        colorscale='RdBu_r',
        zmin=vmin,
        zmax=vmax,
        colorbar=dict(
            title=selected_property,
            titleside='right',
            tickvals=[vmin, vmax],
            ticktext=[f"{vmin:.2f}", f"{vmax:.2f}"]  # 小数点2桁に設定
        )
    ))
    
    # レイアウト設定を更新
    fig.update_layout(
        title=dict(
            text=f"{selected_property}の2次元マップ",
            font=dict(size=26, weight='bold')  # タイトルフォントサイズを26ptに
        ),
        xaxis=dict(
            title=dict(
                text=sweep_elements[0],
                font=dict(size=20, weight='bold')  # X軸ラベルフォントサイズを20ptに
            ),
            tickfont=dict(size=16),  # X軸目盛りフォントサイズを16ptに
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            showline=False,
        ),
        yaxis=dict(
            title=dict(
                text=sweep_elements[1],
                font=dict(size=20, weight='bold')  # Y軸ラベルフォントサイズを20ptに
            ),
            tickfont=dict(size=16),  # Y軸目盛りフォントサイズを16ptに
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            showline=False,
        ),
        width=600,
        height=600,
        plot_bgcolor='white'  # 背景色を白に設定
    )
    
    st.plotly_chart(fig)
    
    # 数値データの表示オプション
    if st.checkbox("数値データを表示"):
        st.write(df[[sweep_elements[0], sweep_elements[1], selected_property]])
    
    # 統計情報の表示
    if st.checkbox("統計情報を表示"):
        st.write("統計情報:")
        stats = df[selected_property].describe()
        st.write(stats)

# メイン処理の前に、必要なディレクトリを作成
def create_required_directories():
    required_dirs = ['mat_model', 'mat_data', 'predict']
    for dir_name in required_dirs:
        os.makedirs(dir_name, exist_ok=True)

# メイン処理
def main():
    create_required_directories()  # 必要なディレクトリを作成
    
    if not check_authentication():
        return
        
    st.title("水素吸蔵合金特性予測システム")
    
    menu = st.sidebar.selectbox(
        "メニュー選択",
        ["学習モデルの作成と管理", "特性予測の実行", "結果の表示と分析"]
    )
    
    if menu == "学習モデルの作成と管理":
        show_model_management()
    elif menu == "特性予測の実行":
        show_prediction()
    else:
        show_analysis()

    if st.button('リロード'):
        st.rerun()




class AlloyNN(nn.Module):
    def __init__(self, input_size):
        super(AlloyNN, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

class AlloyModelTrainer:
    def __init__(self):
        self.input_columns = ['La', 'Ce', 'Pr', 'Nd', 'Sm', 'Y', 'Ca', 'Zr', 
                            'Mg', 'Ni', 'Co', 'Al', 'Mn', 'Zn', 'Cr']
        self.target_columns = ['Temp', 'cap', 'iziritsu', 'density', 'Pd80_04', 'Pd80_05']
        self.scalers = {}
        
    def train_models(self, data_path):
        df = pd.read_csv(data_path)
        X = df[self.input_columns].values
        
        models = {}
        # 入力データのスケーリング
        self.scalers['input'] = MinMaxScaler()
        X_scaled = self.scalers['input'].fit_transform(X)
        
        # 各特性値についてモデルを学習
        for target in self.target_columns:
            y = df[target].values.reshape(-1, 1)
            self.scalers[target] = MinMaxScaler()
            y_scaled = self.scalers[target].fit_transform(y)
            
            model = AlloyNN(len(self.input_columns))
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            # 学習データをTensorに変換
            X_tensor = torch.FloatTensor(X_scaled)
            y_tensor = torch.FloatTensor(y_scaled)
            
            # 学習実行
            for epoch in range(1000):
                optimizer.zero_grad()
                outputs = model(X_tensor)
                loss = criterion(outputs, y_tensor)
                loss.backward()
                optimizer.step()
            
            models[target] = model
            
        return models, self.scalers

def predict_properties(model_path, input_data, scalers):
    # 入力データのスケーリング
    input_scaled = scalers['input'].transform(input_data)
    input_tensor = torch.FloatTensor(input_scaled)
    
    # 結果を格納する辞書
    results = {}
    
    # 各特性値について予測を実行
    properties = ['Temp', 'cap', 'iziritsu', 'density', 'Pd80_04', 'Pd80_05']
    for prop in properties:
        # モデルの読み込み
        model = AlloyNN(input_tensor.shape[1])
        try:
            # モデルの読み込み方法を修正
            state_dict = torch.load(f"{model_path}/{prop}_model.pth", map_location='cpu',weights_only=True)
            model.load_state_dict(state_dict)
        except Exception as e:
            st.error(f"モデル {prop} の読み込み中にエラーが発生しました: {str(e)}")
            continue
            
        model.eval()
        
        # 予測実行
        with torch.no_grad():
            output_scaled = model(input_tensor)
            
        # スケーリングを戻す
        output = scalers[prop].inverse_transform(output_scaled.numpy())
        predicted_value = output[0][0]
        
        # 特性値の制限を適用
        if predicted_value <= 0:
            predicted_value = 0
        elif prop == 'iziritsu' and predicted_value > 1.0:
            predicted_value = 1.0
            
        results[prop] = predicted_value
    
    return results

# モデルの保存時の関数も修正が必要な場合
def save_model(model, path):
    torch.save(model.state_dict(), path)

if __name__ == "__main__":
    main()






