# ============================================================
# 걸생 베타 매미 종 동정 모델 3
# Streamlit 웹사이트
# Mel-Spectrogram + CNN
# ============================================================

import streamlit as st
from pathlib import Path
import tempfile
import numpy as np
import librosa
import tensorflow as tf


# ============================================================
# 1. 기본 설정
# ============================================================

MODEL_NAME = "걸생 베타 매미 종 동정 모델"

CLASS_NAMES = [
    "말매미",
    "참매미",
    "쓰름매미",
    "유지매미",
    "털매미",
    "애매미",
    "배경소리"
]

SAMPLE_RATE = 22050
SEGMENT_SECONDS = 3

N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512


# ============================================================
# 2. 페이지 설정
# ============================================================

st.set_page_config(
    page_title="걸생 베타 매미 종 동정",
    page_icon="🦗",
    layout="centered"
)


# ============================================================
# 3. 제목
# ============================================================

st.title("🦗 걸생 베타 매미 종 동정")

st.write(
    "매미 울음소리를 이용하여 "
    "6종의 매미와 배경소리를 동정합니다."
)

st.divider()


# ============================================================
# 4. 모델 위치
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "model"
    / "매미모델3.keras"
)


# ============================================================
# 5. TensorFlow 설정
# ============================================================

try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception:
    pass


# ============================================================
# 6. 모델 불러오기
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"모델 파일을 찾을 수 없습니다.\n\n"
            f"필요한 위치:\n{MODEL_PATH}"
        )

    return tf.keras.models.load_model(
        MODEL_PATH
    )


# ============================================================
# 7. Mel-Spectrogram 생성
# ============================================================

def make_mel_spectrogram(audio):

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=50,
        fmax=SAMPLE_RATE // 2
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    # 학습할 때와 동일한 정규화
    mel_db = (
        mel_db + 80
    ) / 80

    return mel_db.astype(
        np.float32
    )


# ============================================================
# 8. 음원 분석
# ============================================================

def predict_audio(audio_path):

    model = load_model()

    # --------------------------------------------------------
    # 음원 불러오기
    # --------------------------------------------------------

    audio, _ = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    if len(audio) == 0:

        raise ValueError(
            "분석할 수 있는 음원이 없습니다."
        )

    # --------------------------------------------------------
    # 3초 길이
    # --------------------------------------------------------

    segment_samples = (
        SAMPLE_RATE
        * SEGMENT_SECONDS
    )

    probability_sum = np.zeros(
        len(CLASS_NAMES),
        dtype=np.float32
    )

    analyzed_count = 0

    # --------------------------------------------------------
    # 3초씩 분석
    # --------------------------------------------------------

    for start in range(
        0,
        len(audio),
        segment_samples
    ):

        segment = audio[
            start:
            start + segment_samples
        ]

        # 마지막 구간이 3초보다 짧으면 0으로 채움
        if len(segment) < segment_samples:

            segment = np.pad(
                segment,
                (
                    0,
                    segment_samples
                    - len(segment)
                )
            )

        # ----------------------------------------------------
        # Mel-Spectrogram
        # ----------------------------------------------------

        mel = make_mel_spectrogram(
            segment
        )

        # CNN 입력 형태
        x_one = mel[
            np.newaxis,
            ...,
            np.newaxis
        ]

        # ----------------------------------------------------
        # 예측
        # ----------------------------------------------------

        prediction = model.predict(
            x_one,
            verbose=0
        )[0]

        probability_sum += prediction

        analyzed_count += 1

    # --------------------------------------------------------
    # 평균 확률
    # --------------------------------------------------------

    if analyzed_count == 0:

        raise ValueError(
            "분석할 수 있는 음원이 없습니다."
        )

    average_prediction = (
        probability_sum
        / analyzed_count
    )

    percentages = (
        average_prediction * 100
    )

    best_index = int(
        np.argmax(
            average_prediction
        )
    )

    return {
        "best_species":
            CLASS_NAMES[best_index],

        "best_percentage":
            float(
                percentages[best_index]
            ),

        "probabilities": [
            {
                "species": species,
                "percentage":
                    float(percentage)
            }

            for species, percentage
            in zip(
                CLASS_NAMES,
                percentages
            )
        ],

        "segment_count":
            analyzed_count,

        "audio_length":
            len(audio)
            / SAMPLE_RATE
    }


# ============================================================
# 9. 음원 업로드
# ============================================================

uploaded_file = st.file_uploader(
    "매미 울음소리 파일을 업로드하세요.",
    type=[
        "mp3",
        "wav",
        "m4a",
        "flac"
    ]
)


# ============================================================
# 10. 업로드된 음원 표시
# ============================================================

if uploaded_file is not None:

    st.success(
        f"파일 업로드 완료: {uploaded_file.name}"
    )

    st.audio(
        uploaded_file
    )

    st.divider()


# ============================================================
# 11. 동정 시작
# ============================================================

if uploaded_file is not None:

    if st.button(
        "🔎 매미 종 동정 시작",
        use_container_width=True
    ):

        with st.spinner(
            "매미 울음소리를 분석하고 있습니다..."
        ):

            try:

                # 임시 파일 생성
                suffix = Path(
                    uploaded_file.name
                ).suffix

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix
                ) as temp_file:

                    temp_file.write(
                        uploaded_file.getbuffer()
                    )

                    temp_path = (
                        temp_file.name
                    )

                # 분석
                result = predict_audio(
                    temp_path
                )

                # 임시 파일 삭제
                Path(
                    temp_path
                ).unlink(
                    missing_ok=True
                )

                # 결과 저장
                st.session_state[
                    "result"
                ] = result

            except Exception as e:

                st.error(
                    "음원 분석 중 오류가 발생했습니다."
                )

                st.exception(e)


# ============================================================
# 12. 결과 표시
# ============================================================

if "result" in st.session_state:

    result = st.session_state[
        "result"
    ]

    st.divider()

    st.subheader(
        "🧬 동정 결과"
    )

    # --------------------------------------------------------
    # 가장 높은 확률
    # --------------------------------------------------------

    st.success(
        f"가장 높은 예측: "
        f"**{result['best_species']}**"
    )

    st.metric(
        "예측 확률",
        f"{result['best_percentage']:.2f}%"
    )

    # --------------------------------------------------------
    # 음원 정보
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "음원 길이",
            f"{result['audio_length']:.2f}초"
        )

    with col2:

        st.metric(
            "분석 구간",
            f"{result['segment_count']}개"
        )

    st.divider()

    # --------------------------------------------------------
    # 종별 확률
    # --------------------------------------------------------

    st.subheader(
        "종별 예측 확률"
    )

    for item in result[
        "probabilities"
    ]:

        species = item[
            "species"
        ]

        percentage = item[
            "percentage"
        ]

        st.write(
            f"**{species}** "
            f"{percentage:.2f}%"
        )

        st.progress(
            min(
                max(
                    percentage / 100,
                    0.0
                ),
                1.0
            )
        )


# ============================================================
# 13. 안내
# ============================================================

st.divider()

st.caption(
    "걸생 베타 매미 종 동정 모델 3 · "
    "Mel-Spectrogram + CNN"
)
