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

# 認証用の関数を更新
def verify_credentials(username, password):
    valid_credentials_admin = {
        "kai_adm": "pass_adm",
        "cae_adm": "cae_adm"
    }
    valid_credentials_user = {
        "kai01": "pass_kai",
        "cae01": "pass_cae"
    }
    
    # 管理者認証
    if username in valid_credentials_admin and valid_credentials_admin[username] == password:
        return "admin"
    # 一般ユーザー認証
    elif username in valid_credentials_user and valid_credentials_user[username] == password:
        return "user"
    # 認証失敗
    return None

def check_authentication():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.user_role = None
        
    if not st.session_state.authenticated:
        username = st.text_input("ユーザー名")
        password = st.text_input("パスワード", type="password")
        
        if st.button("ログイン"):
            role = verify_credentials(username, password)
            if role:
                st.session_state.authenticated = True
                st.session_state.user_role = role
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
    
    # 元素リストの定義を関数の先頭に移動
    elements = ['La', 'Ce', 'Pr', 'Nd', 'Sm', 'Y', 'Ca', 'Zr', 
               'Mg', 'Ni', 'Co', 'Al', 'Mn', 'Zn', 'Cr']
    re_elements = ['La', 'Ce', 'Pr', 'Nd', 'Sm', 'Y', 'Zr', 'Mg']
    
    # 組成設定の保存/読み込みセクション
    st.subheader("組成設定の保存/読み込み")
    col1, col2 = st.columns(2)
    
    # スイープ範囲の定義（デフォルト値）
    if 'sweep_ranges' not in st.session_state:
        st.session_state.sweep_ranges = {
            'La': (0.0, 0.5), 'Ce': (0.0, 0.5), 'Pr': (0.0, 0.5),
            'Nd': (0.0, 0.5), 'Sm': (0.0, 0.5), 'Y': (0.0, 0.5),
            'Ca': (0.0, 0.001), 'Zr': (0.0, 0.5), 'Mg': (0.0, 0.5),
            'Ni': (0.0, 0.5), 'Co': (0.0, 0.0), 'Al': (0.0, 0.4),
            'Mn': (0.0, 0.5), 'Zn': (0.0, 0.3), 'Cr': (0.0, 0.1)
        }
    
    with col1:
        # 保存済み組成ファイルの表示
        composition_files = [f for f in os.listdir("mat_model") if f.endswith("_composition.csv")]
        if composition_files:
            #st.write("保存済み組成設定:")
            selected_composition = st.selectbox("保存済みの設定を選択", composition_files)
            
            # ロードと削除のボタンを横に並べる
            load_col, delete_col = st.columns(2)
            
            with load_col:
                if st.button("組成をロード"):
                    try:
                        df = pd.read_csv(f"mat_model/{selected_composition}")
                        # 基本の組成値を読み込み
                        composition_data = df.iloc[0].to_dict()
                        
                        # スイープ設定を読み込み
                        sweep_elements = []
                        if 'sweep_element1' in composition_data and pd.notna(composition_data['sweep_element1']):
                            sweep_elements.append(composition_data['sweep_element1'])
                        if 'sweep_element2' in composition_data and pd.notna(composition_data['sweep_element2']):
                            sweep_elements.append(composition_data['sweep_element2'])
                        
                        # スイープ範囲を読み込み
                        sweep_ranges = st.session_state.sweep_ranges.copy()
                        for elem in sweep_elements:
                            if f'{elem}_start' in composition_data and f'{elem}_end' in composition_data:
                                sweep_ranges[elem] = (
                                    float(composition_data[f'{elem}_start']),
                                    float(composition_data[f'{elem}_end'])
                                )
                        
                        # セッション状態を更新
                        st.session_state.composition_values = {k: v for k, v in composition_data.items() 
                                                             if k in elements}
                        st.session_state.sweep_elements = sweep_elements
                        st.session_state.sweep_ranges = sweep_ranges
                        st.session_state.adjust_element = composition_data.get('adjust_element', '')
                        
                        st.success("組成設定をロードしました")
                        st.rerun()
                    except Exception as e:
                        st.error(f"組成設定のロード中にエラーが発生しました: {str(e)}")
            
            with delete_col:
                if st.button("設定を削除"):
                    st.session_state.show_delete_confirmation = True
                    st.session_state.file_to_delete = selected_composition
                
                # 削除確認のダイアログを表示
                if st.session_state.get('show_delete_confirmation', False):
                    st.warning(f"本当に「{st.session_state.file_to_delete}」を削除しますか？")
                    if st.button("削除を確定"):
                        try:
                            os.remove(f"mat_model/{st.session_state.file_to_delete}")
                            st.success(f"組成設定「{st.session_state.file_to_delete}」を削除しました")
                            # 確認ダイアログをクリア
                            st.session_state.show_delete_confirmation = False
                            st.session_state.file_to_delete = None
                            st.rerun()
                        except Exception as e:
                            st.error(f"削除中にエラーが発生しました: {str(e)}")
                    if st.button("キャンセル"):
                        st.session_state.show_delete_confirmation = False
                        st.session_state.file_to_delete = None
                        st.rerun()
    
    with col2:
        # 新規組成の保存
        #st.write("新規保存:")
        composition_name = st.text_input("組成設定を新規作成", "composition_name")
        if st.button("現在の組成を保存"):
            if 'composition_values' in st.session_state:
                try:
                    # 基本の組成値
                    save_data = st.session_state.composition_values.copy()
                    
                    # スイープ設定を追加
                    sweep_elements = st.session_state.get('sweep_elements', [])
                    save_data['sweep_element1'] = sweep_elements[0] if len(sweep_elements) > 0 else None
                    save_data['sweep_element2'] = sweep_elements[1] if len(sweep_elements) > 1 else None
                    
                    # スイープ範囲を追加
                    for elem in sweep_elements:
                        start, end = st.session_state.sweep_ranges[elem]
                        save_data[f'{elem}_start'] = start
                        save_data[f'{elem}_end'] = end
                    
                    # 調整元素を追加
                    save_data['adjust_element'] = st.session_state.get('adjust_element', '')
                    
                    # データフレームに変換して保存
                    df = pd.DataFrame([save_data])
                    df.to_csv(f"mat_model/{composition_name}_composition.csv", index=False)
                    st.success("組成設定を保存しました")
                    st.rerun()
                except Exception as e:
                    st.error(f"組成設定の保存中にエラーが発生しました: {str(e)}")
            else:
                st.warning("保存する組成値がありません")
    
    # モデル選択
    st.subheader("予測モデルの選択")
    model_files = [f for f in os.listdir("mat_model") if not f.endswith("_composition.csv")]
    if not model_files:
        st.error("利用可能なモデルがありません。先にモデルを作成してください。")
        return
    
    selected_model = st.selectbox("予測に使用するモデル", model_files)
    
    # 元素リストの定義
    elements = ['La', 'Ce', 'Pr', 'Nd', 'Sm', 'Y', 'Ca', 'Zr', 
               'Mg', 'Ni', 'Co', 'Al', 'Mn', 'Zn', 'Cr']
    re_elements = ['La', 'Ce', 'Pr', 'Nd', 'Sm', 'Y', 'Zr', 'Mg']
    
    # スイープする元素の選択
    sweep_elements = st.multiselect("スイープする元素を選択（最大2つ）", 
                                  elements, 
                                  default=st.session_state.get('sweep_elements', []),
                                  max_selections=2)
    st.session_state.sweep_elements = sweep_elements
    
    # スイープ範囲の設定
    if sweep_elements:
        st.subheader("スイープ範囲の設定")
        sweep_ranges = st.session_state.sweep_ranges.copy()
        
        for elem in sweep_elements:
            col1, col2 = st.columns(2)
            with col1:
                start = st.number_input(f"{elem}のスタート値",
                                      value=float(sweep_ranges[elem][0]),
                                      format="%.3f")
            with col2:
                end = st.number_input(f"{elem}のエンド値",
                                    value=float(sweep_ranges[elem][1]),
                                    format="%.3f")
            sweep_ranges[elem] = (start, end)
        
        st.session_state.sweep_ranges = sweep_ranges
    
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
                default_value = st.session_state.composition_values.get(elem, 0.0) if 'composition_values' in st.session_state else 0.0
                input_values[elem] = st.number_input(f"{elem}の値",
                                                   value=default_value,
                                                   format="%.2f")
    
    # セッションステートに現在の組成値を保存
    st.session_state.composition_values = input_values
    
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

    # ファイル管理セクションの更新
    st.markdown("---")  # 区切り線
    st.subheader("予測結果ファイルの保存と削除")
    file_to_manage = st.selectbox("ファイルを選択", predict_files, key="manage_file")
    
    # ダウンロードと削除のボタンを横に並べる
    col1, col2 = st.columns(2)
    
    with col1:
        # ファイルの内容を読み込んでダウンロードボタンを作成
        try:
            df = pd.read_csv(f"predict/{file_to_manage}")
            csv = df.to_csv(index=False)
            st.download_button(
                label="ファイルをダウンロード",
                data=csv,
                file_name=file_to_manage,
                mime='text/csv'
            )
        except Exception as e:
            st.error(f"ファイルの読み込み中にエラーが発生しました: {str(e)}")
    
    with col2:
        if st.button("ファイルを削除"):
            st.session_state.show_delete_confirmation = True
            st.session_state.file_to_delete = file_to_manage
        
        # 削除確認のダイアログを表示
        if st.session_state.get('show_delete_confirmation', False):
            st.warning(f"本当に「{st.session_state.file_to_delete}」を削除しますか？")
            if st.button("削除を確定"):
                try:
                    os.remove(f"predict/{st.session_state.file_to_delete}")
                    st.success(f"ファイル {st.session_state.file_to_delete} を削除しました")
                    # 確認ダイアログをクリア
                    st.session_state.show_delete_confirmation = False
                    st.session_state.file_to_delete = None
                    st.rerun()
                except Exception as e:
                    st.error(f"削除中にエラーが発生しました: {str(e)}")
            if st.button("キャンセル"):
                st.session_state.show_delete_confirmation = False
                st.session_state.file_to_delete = None
                st.rerun()

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
#            titleside='right',
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

def show_data_preparation():
    st.header("学習データの作成")
    
    # exp_dataフォルダからマスターデータのリストを取得
    exp_files = os.listdir("exp_data")
    if not exp_files:
        st.error("マスターデータが存在しません。exp_dataフォルダにデータを配置してください。")
        return
    
    # ファイル選択
    selected_file = st.selectbox("マスターデータの選択", exp_files)
    
    if st.button("データ補完を実行"):
        with st.spinner("データ補完中..."):
            try:
                # マスターデータの読み込み
                df = pd.read_csv(f"exp_data/{selected_file}")
                
                # 入力特性と目標特性の定義
                input_columns = ['La', 'Ce', 'Pr', 'Nd', 'Sm', 'Y', 'Ca', 'Zr', 
                               'Mg', 'Ni', 'Co', 'Al', 'Mn', 'Zn', 'Cr']
                target_columns = ['Temp', 'cap', 'iziritsu', 'density', 'Pd80_04', 'Pd80_05']
                
                # 各特性について欠損値を補完
                for target in target_columns:
                    # 欠損値が存在する場合のみ処理
                    if df[target].isnull().any():
                        # 学習データとテストデータの分離
                        train_data = df[~df[target].isnull()]
                        test_data = df[df[target].isnull()]
                        
                        # スケーラーの初期化と適用
                        input_scaler = MinMaxScaler()
                        target_scaler = MinMaxScaler()
                        
                        # 学習データの準備
                        X_train = input_scaler.fit_transform(train_data[input_columns])
                        y_train = target_scaler.fit_transform(train_data[[target]])
                        
                        # モデルの定義と学習
                        model = AlloyNN(len(input_columns))
                        criterion = nn.MSELoss()
                        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                        
                        # テンソルに変換
                        X_train_tensor = torch.FloatTensor(X_train)
                        y_train_tensor = torch.FloatTensor(y_train)
                        
                        # 学習の実行
                        for epoch in range(1000):
                            optimizer.zero_grad()
                            outputs = model(X_train_tensor)
                            loss = criterion(outputs, y_train_tensor)
                            loss.backward()
                            optimizer.step()
                        
                        # 欠損値の予測
                        if len(test_data) > 0:
                            X_test = input_scaler.transform(test_data[input_columns])
                            X_test_tensor = torch.FloatTensor(X_test)
                            
                            model.eval()
                            with torch.no_grad():
                                y_pred = model(X_test_tensor)
                                y_pred = target_scaler.inverse_transform(y_pred.numpy())
                            
                            # 予測値で欠損値を補完
                            df.loc[df[target].isnull(), target] = y_pred.flatten()
                
                # 補完したデータの保存
                output_filename = selected_file.replace('.csv', '_calculated.csv')
                df.to_csv(f"mat_data/{output_filename}", index=False)
                st.success(f"データ補完が完了しました。保存先: mat_data/{output_filename}")
                
                # 補完前後のデータ比較を表示
                st.subheader("補完結果の確認")
                original_df = pd.read_csv(f"exp_data/{selected_file}")
                
                for target in target_columns:
                    if original_df[target].isnull().any():
                        st.write(f"### {target}の補完結果")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("補完前の欠損値数:", original_df[target].isnull().sum())
                        with col2:
                            st.write("補完後の欠損値数:", df[target].isnull().sum())
                
            except Exception as e:
                st.error(f"データ補完中にエラーが発生しました: {str(e)}")

# 必要なディレクトリの作成関数を追加
def create_required_directories():
    required_dirs = ['mat_model', 'mat_data', 'predict', 'exp_data']
    for dir_name in required_dirs:
        os.makedirs(dir_name, exist_ok=True)

def main():
    create_required_directories()  # 必要なディレクトリを作成
    
    if not check_authentication():
        return
        
    st.title("水素吸蔵合金特性予測システム")
    
    # ユーザーの権限に応じてメニュー項目を設定
    if st.session_state.user_role == "admin":
        menu_items = ["学習データの作成", "学習モデルの作成と管理", "特性予測の実行", "結果の表示と分析"]
    else:  # user
        menu_items = ["学習モデルの作成と管理", "特性予測の実行", "結果の表示と分析"]
    
    menu = st.sidebar.selectbox(
        "メニュー選択",
        menu_items
    )
    
    if menu == "学習データの作成":
        if st.session_state.user_role == "admin":
            show_data_preparation()
        else:
            st.error("このメニューにアクセスする権限がありません")
    elif menu == "学習モデルの作成と管理":
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






