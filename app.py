import streamlit as st
from pathlib import Path
import tempfile
import os
import numpy as np
import librosa
import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

MODEL_NAME = "걸생 베타 매미 종 동정 모델"
CLASS_NAMES = ["말매미", "참매미", "쓰름매미", "유지매미", "털매미", "애매미", "배경소리"]
SAMPLE_RATE = 22050
SEGMENT_SECONDS = 3
N_MFCC = 40

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "model.keras"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

def predict_audio(audio_path):
    model = load_model()
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True, duration=180)
    segment_samples = SAMPLE_RATE * SEGMENT_SECONDS
    probability_sum = np.zeros(len(CLASS_NAMES), dtype=np.float32)
    analyzed_count = 0

    for start in range(0, len(audio), segment_samples):
        segment = audio[start:start + segment_samples]
        if len(segment) < segment_samples:
            segment = np.pad(segment, (0, segment_samples - len(segment)))

        mfcc = librosa.feature.mfcc(y=segment, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
        mfcc_mean = np.mean(mfcc, axis=1)
        x_one = np.expand_dims(mfcc_mean, axis=0)
        prediction = model.predict(x_one, verbose=0)[0]

        probability_sum += prediction
        analyzed_count += 1

        del segment, mfcc, mfcc_mean, x_one, prediction

    if analyzed_count == 0:
        raise ValueError("분석할 수 있는 음원이 없습니다.")

    average_prediction = probability_sum / analyzed_count
    percentages = average_prediction * 100
    best_index = int(np.argmax(average_prediction))

    return {
        "best_species": CLASS_NAMES[best_index],
        "best_percentage": float(percentages[best_index]),
        "probabilities": [
            {"species": s, "percentage": float(p)}
            for s, p in zip(CLASS_NAMES, percentages)
        ],
        "segment_count": analyzed_count,
        "audio_length": len(audio) / SAMPLE_RATE
    }

st.set_page_config(page_title=MODEL_NAME, page_icon="🦗", layout="centered")

st.markdown("""
<style>
.main-title{text-align:center;font-size:2.2rem;font-weight:700;margin-bottom:.3rem}
.subtitle{text-align:center;color:#607064;margin-bottom:2rem}
.result-box{padding:1.5rem;border-radius:18px;background:#f1f7f2;text-align:center;margin:1rem 0}
.result-species{font-size:2rem;font-weight:700}
.result-percent{font-size:1.2rem;margin-top:.5rem}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">걸생 베타 매미 종 동정</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">매미 울음소리를 이용하여 6종의 매미와 배경소리를 동정합니다.</div>', unsafe_allow_html=True)

st.info("MP3, WAV, M4A, FLAC 파일을 올릴 수 있습니다.")

uploaded_file = st.file_uploader("울음소리 파일을 선택하세요", type=["mp3", "wav", "m4a", "flac"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    if st.button("🔎 매미 종 동정 시작", use_container_width=True):
        suffix = Path(uploaded_file.name).suffix.lower()
        temp_path = None

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_path = temp_file.name

            with st.spinner("음원을 분석하고 있습니다..."):
                result = predict_audio(temp_path)

            st.success("동정이 완료되었습니다!")

            st.markdown(
                f'<div class="result-box"><div>가장 높은 확률의 종</div>'
                f'<div class="result-species">{result["best_species"]}</div>'
                f'<div class="result-percent">{result["best_percentage"]:.2f}%</div></div>',
                unsafe_allow_html=True
            )

            st.write(f'분석 음원 길이: {result["audio_length"]:.2f}초')
            st.write(f'분석 구간 수: {result["segment_count"]}개')
            st.subheader("종별 예상 확률")

            for item in result["probabilities"]:
                species = item["species"]
                percentage = item["percentage"]
                st.write(f"**{species}** — {percentage:.2f}%")
                st.progress(min(percentage / 100, 1.0))

        except Exception as e:
            st.error("음원 분석 중 오류가 발생했습니다.")
            st.code(str(e))

        finally:
            if temp_path is not None and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

st.divider()
st.caption("걸생 베타 매미 종 동정 모델 · MFCC 기반 음향 분류")
