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
    
    # ファイルアップロード（複数ファイル対応）
    uploaded_files = st.file_uploader("音声ファイルをアップロード（複数選択可）", 
                                     type=["mp3", "wav", "m4a", "ogg", "flac"],
                                     accept_multiple_files=True,
                                     help="対応フォーマット: MP3, WAV, M4A, OGG, FLAC")
    
    if uploaded_files:
        # ファイル数と合計サイズの表示
        total_size = sum(file.size for file in uploaded_files) / (1024 * 1024)
        st.info(f"選択されたファイル数: {len(uploaded_files)} (合計サイズ: {total_size:.2f} MB)")
        
        # 文字起こし実行ボタン
        transcribe_button = st.button("文字起こし開始", type="primary")
        
        if transcribe_button:
            # モデルロード（一度だけ）
            with st.spinner("モデルをロード中..."):
                model_load_start = time.time()
                model = load_whisper_model(model_option)
                model_load_time = time.time() - model_load_start
                st.success(f"モデルロード完了（{model_load_time:.2f}秒）")
            
            # 進捗バー
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 結果を保存するリスト
            all_results = []
            all_timestamps = []
            
            # 各ファイルを処理
            for idx, uploaded_file in enumerate(uploaded_files):
                file_name = uploaded_file.name
                file_size_mb = uploaded_file.size / (1024 * 1024)
                
                status_text.text(f"処理中: {file_name} ({idx + 1}/{len(uploaded_files)})")
                progress_bar.progress((idx + 1) / len(uploaded_files))
                
                # 一時ファイルとして保存
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_filename = tmp_file.name
                
                try:
                    # 文字起こし処理
                    transcribe_start = time.time()
                    
                    # 言語オプション設定
                    options = {}
                    if language_option:
                        options["language"] = language_option
                    
                    # 文字起こし実行
                    result = model.transcribe(temp_filename, **options)
                    
                    transcribe_time = time.time() - transcribe_start
                    
                    # 結果を保存
                    file_result = {
                        "filename": file_name,
                        "text": result["text"],
                        "transcribe_time": transcribe_time,
                        "segments": result["segments"]
                    }
                    all_results.append(file_result)
                    
                    # タイムスタンプ付きテキストも保存
                    timestamp_text = ""
                    for segment in result["segments"]:
                        start_time = segment["start"]
                        end_time = segment["end"]
                        text = segment["text"]
                        
                        start_formatted = str(datetime.utcfromtimestamp(start_time).strftime('%H:%M:%S.%f'))[:-3]
                        end_formatted = str(datetime.utcfromtimestamp(end_time).strftime('%H:%M:%S.%f'))[:-3]
                        
                        timestamp_text += f"[{start_formatted} --> {end_formatted}] {text}\n"
                    
                    all_timestamps.append({
                        "filename": file_name,
                        "timestamp_text": timestamp_text
                    })
                
                except Exception as e:
                    st.error(f"エラーが発生しました ({file_name}): {str(e)}")
                
                finally:
                    # 一時ファイルの削除
                    if os.path.exists(temp_filename):
                        os.unlink(temp_filename)
            
            # 進捗完了
            progress_bar.progress(1.0)
            status_text.text("処理完了！")
            
            # 結果の表示
            st.markdown("### 文字起こし結果")
            
            # タブで各ファイルの結果を表示
            tabs = st.tabs([f"📄 {result['filename']}" for result in all_results])
            
            for tab, result in zip(tabs, all_results):
                with tab:
                    st.success(f"処理時間: {result['transcribe_time']:.2f}秒")
                    
                    # 音声再生
                    for uploaded_file in uploaded_files:
                        if uploaded_file.name == result["filename"]:
                            st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")
                            break
                    
                    # テキスト結果表示
                    st.text_area(f"{result['filename']} のテキスト", value=result["text"], height=200, key=f"text_{result['filename']}")
                    
                    # 個別ダウンロードボタン
                    st.download_button(
                        label=f"{result['filename']} をダウンロード",
                        data=result["text"],
                        file_name=f"{os.path.splitext(result['filename'])[0]}_transcript.txt",
                        mime="text/plain",
                        key=f"download_{result['filename']}"
                    )
                    
                    # タイムスタンプ付きの詳細結果
                    with st.expander("詳細（タイムスタンプ付き）"):
                        # タイムスタンプ付きテキストを検索
                        timestamp_data = next((t for t in all_timestamps if t["filename"] == result["filename"]), None)
                        if timestamp_data:
                            st.text_area("タイムスタンプ付きテキスト", 
                                       value=timestamp_data["timestamp_text"], 
                                       height=200, 
                                       key=f"timestamp_{result['filename']}")
                            
                            st.download_button(
                                label="タイムスタンプ付きテキストをダウンロード",
                                data=timestamp_data["timestamp_text"],
                                file_name=f"{os.path.splitext(result['filename'])[0]}_transcript_timestamps.txt",
                                mime="text/plain",
                                key=f"download_timestamp_{result['filename']}"
                            )
            
            # 全ファイルまとめてダウンロード
            if len(all_results) > 1:
                st.markdown("---")
                st.markdown("### 全ファイルまとめてダウンロード")
                
                # 全テキストを結合
                combined_text = ""
                combined_timestamp_text = ""
                
                for result in all_results:
                    combined_text += f"=== {result['filename']} ===\n\n"
                    combined_text += result["text"] + "\n\n"
                    
                    timestamp_data = next((t for t in all_timestamps if t["filename"] == result["filename"]), None)
                    if timestamp_data:
                        combined_timestamp_text += f"=== {result['filename']} ===\n\n"
                        combined_timestamp_text += timestamp_data["timestamp_text"] + "\n\n"
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.download_button(
                        label="全テキストをダウンロード",
                        data=combined_text,
                        file_name="all_transcripts.txt",
                        mime="text/plain"
                    )
                
                with col2:
                    st.download_button(
                        label="全タイムスタンプ付きテキストをダウンロード",
                        data=combined_timestamp_text,
                        file_name="all_transcripts_timestamps.txt",
                        mime="text/plain"
                    )
    
    else:
        # ファイルがアップロードされていない場合の表示
        st.info("👆 音声ファイルをアップロードしてください")
        
        # サンプル説明
        with st.expander("使い方"):
            st.markdown("""
            1. サイドバーでモデルサイズと言語を選択
            2. 音声ファイルをアップロード（複数ファイル選択可）
            3. 「文字起こし開始」ボタンをクリック
            4. 各ファイルの結果を確認し、必要に応じてダウンロード
            5. 複数ファイルの場合、まとめてダウンロードも可能
            
            **モデルサイズについて:**
            - tiny: 最小・最速（低精度）
            - base: バランス型（推奨）
            - small: 中程度の精度
            - medium: 高精度
            - large: 最高精度（処理時間が長い）
            
            **複数ファイルの処理について:**
            - 同時に複数のファイルを選択できます
            - 各ファイルは順番に処理されます
            - 結果はタブで切り替えて確認できます
            - まとめてダウンロードも可能です
            """)

if __name__ == "__main__":
    main()