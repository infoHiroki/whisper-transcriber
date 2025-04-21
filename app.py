#!/usr/bin/env python3
"""
Whisper文字起こしWebアプリ（Streamlit使用）
"""

import os
import sys
import time
import tempfile
import whisper
import torch
import streamlit as st
from datetime import datetime
import subprocess

# ページ設定
st.set_page_config(
    page_title="Whisper文字起こしツール",
    page_icon="🎤",
    layout="wide"
)

# キャッシュ設定（モデルを再ロードしないようにする）
@st.cache_resource
def load_whisper_model(model_name):
    """Whisperモデルをロードする（キャッシュ使用）"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return whisper.load_model(model_name, device=device)

def check_ffmpeg():
    """FFmpegがインストールされているか確認"""
    try:
        # まず通常のパスでチェック
        result = subprocess.run(['ffmpeg', '-version'], 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE, 
                              text=True)
        
        if result.returncode == 0:
            st.sidebar.success("✅ FFmpegが検出されました。")
            return True
        
        # 通常のパスで見つからない場合、Chocolateyパスを確認
        chocolatey_path = r'C:\ProgramData\chocolatey\bin\ffmpeg.exe'
        if os.path.exists(chocolatey_path):
            try:
                result = subprocess.run([chocolatey_path, '-version'],
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE,
                                      text=True)
                if result.returncode == 0:
                    st.sidebar.success(f"✅ FFmpegが検出されました (Chocolatey: {chocolatey_path})")
                    # 環境変数を一時的に更新
                    os.environ['PATH'] = r'C:\ProgramData\chocolatey\bin;' + os.environ['PATH']
                    return True
            except Exception:
                pass
        
        st.warning("⚠️ FFmpegが見つかりませんでした。音声ファイルの処理で問題が発生する可能性があります。")
        st.info("💡 FFmpegがインストールされている場合は、PowerShellまたはコマンドプロンプトを新しく開いて再度実行してください。")
        # 警告は表示するが、処理は続行
        return False
    except Exception as e:
        st.warning(f"⚠️ FFmpegのチェック中に問題が発生しました: {str(e)}")
        st.info("💡 FFmpegがインストールされていることを確認し、PATHが正しく設定されているか確認してください。")
        # エラーの場合も処理は続行
        return False

def get_available_models():
    """利用可能なWhisperモデルの一覧を返す"""
    return ["tiny", "base", "small", "medium", "large"]

def main():
    """メイン関数"""
    st.title("🎤 Whisper文字起こしツール")
    st.markdown("""
    OpenAIのWhisperモデルを使用して、音声ファイルからテキストへの文字起こしを行います。
    """)
    
    # FFmpegの確認（エラーで停止しない）
    check_ffmpeg()
    
    # サイドバー設定
    st.sidebar.title("設定")
    
    # モデル選択
    model_option = st.sidebar.selectbox(
        "モデルサイズを選択",
        options=get_available_models(),
        index=1,  # baseをデフォルトに
        help="大きいモデルほど精度が上がりますが、処理時間も増加します。"
    )
    
    # 言語選択
    language_option = st.sidebar.selectbox(
        "言語を選択（自動検出する場合は空欄）",
        options=["", "en", "ja", "zh", "de", "fr", "es", "ko", "ru"],
        index=0,
        format_func=lambda x: {
            "": "自動検出", "en": "英語", "ja": "日本語", "zh": "中国語",
            "de": "ドイツ語", "fr": "フランス語", "es": "スペイン語",
            "ko": "韓国語", "ru": "ロシア語"
        }.get(x, x),
        help="音声の言語を指定します。自動検出も可能です。"
    )
    
    # デバイス情報表示
    device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    st.sidebar.info(f"使用デバイス: {device}")
    
    if device == "CPU":
        st.sidebar.warning("GPUが検出されませんでした。処理が遅くなる可能性があります。")
    
    # サイドバーにGitHubリンク
    st.sidebar.markdown("---")
    st.sidebar.markdown("[GitHubリポジトリ](https://github.com/fumifumi0831/whisper-transcription)")
    
    # ファイルアップロード
    uploaded_file = st.file_uploader("音声ファイルをアップロード", 
                                    type=["mp3", "wav", "m4a", "ogg", "flac"],
                                    help="対応フォーマット: MP3, WAV, M4A, OGG, FLAC")
    
    if uploaded_file is not None:
        # ファイル情報表示
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.info(f"ファイル: {uploaded_file.name} ({file_size_mb:.2f} MB)")
        
        # 音声再生機能
        st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")
        
        # 文字起こし実行ボタン
        transcribe_button = st.button("文字起こし開始", type="primary")
        
        if transcribe_button:
            # 処理開始
            with st.spinner("文字起こし処理中..."):
                # 一時ファイルとして保存
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_filename = tmp_file.name
                
                try:
                    # モデルロード
                    load_start = time.time()
                    progress_text = st.empty()
                    progress_text.text("モデルをロード中...")
                    model = load_whisper_model(model_option)
                    load_end = time.time()
                    progress_text.text(f"モデルロード完了（{load_end - load_start:.2f}秒）")
                    
                    # 文字起こし処理
                    progress_text.text("文字起こし処理中...")
                    transcribe_start = time.time()
                    
                    # 言語オプション設定
                    options = {}
                    if language_option:
                        options["language"] = language_option
                        
                    # 文字起こし実行
                    result = model.transcribe(temp_filename, **options)
                    
                    transcribe_end = time.time()
                    progress_text.empty()
                    
                    # 処理時間計算
                    transcribe_time = transcribe_end - transcribe_start
                    total_time = transcribe_end - load_start
                    
                    # 結果表示
                    st.markdown("### 文字起こし結果")
                    st.success(f"処理完了（文字起こし: {transcribe_time:.2f}秒、合計: {total_time:.2f}秒）")
                    
                    # テキスト結果表示
                    st.markdown("#### テキスト")
                    st.text_area("", value=result["text"], height=200)
                    
                    # ダウンロードボタン
                    st.download_button(
                        label="テキストをダウンロード",
                        data=result["text"],
                        file_name=f"{os.path.splitext(uploaded_file.name)[0]}_transcript.txt",
                        mime="text/plain"
                    )
                    
                    # タイムスタンプ付きの詳細結果
                    with st.expander("詳細（タイムスタンプ付き）"):
                        # テーブル表示用のデータ準備
                        table_data = []
                        timestamp_text = ""
                        
                        for segment in result["segments"]:
                            start_time = segment["start"]
                            end_time = segment["end"]
                            text = segment["text"]
                            
                            # 時間をフォーマット (HH:MM:SS.ms)
                            start_formatted = str(datetime.utcfromtimestamp(start_time).strftime('%H:%M:%S.%f'))[:-3]
                            end_formatted = str(datetime.utcfromtimestamp(end_time).strftime('%H:%M:%S.%f'))[:-3]
                            
                            table_data.append({
                                "開始": start_formatted,
                                "終了": end_formatted,
                                "テキスト": text
                            })
                            
                            timestamp_text += f"[{start_formatted} --> {end_formatted}] {text}\n"
                        
                        # テーブル表示
                        st.table(table_data)
                        
                        # タイムスタンプ付きテキストのダウンロードボタン
                        st.download_button(
                            label="タイムスタンプ付きテキストをダウンロード",
                            data=timestamp_text,
                            file_name=f"{os.path.splitext(uploaded_file.name)[0]}_transcript_timestamps.txt",
                            mime="text/plain"
                        )
                
                except Exception as e:
                    st.error(f"エラーが発生しました: {str(e)}")
                
                finally:
                    # 一時ファイルの削除
                    if os.path.exists(temp_filename):
                        os.unlink(temp_filename)
    
    else:
        # ファイルがアップロードされていない場合の表示
        st.info("👆 音声ファイルをアップロードしてください")
        
        # サンプル説明
        with st.expander("使い方"):
            st.markdown("""
            1. サイドバーでモデルサイズと言語を選択
            2. 音声ファイルをアップロード
            3. 「文字起こし開始」ボタンをクリック
            4. 結果を確認し、必要に応じてダウンロード
            
            **モデルサイズについて:**
            - tiny: 最小・最速（低精度）
            - base: バランス型（推奨）
            - small: 中程度の精度
            - medium: 高精度
            - large: 最高精度（処理時間が長い）
            """)

if __name__ == "__main__":
    main()