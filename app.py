# 걸생 베타 매미 종 동정 모델 - 웹사이트 버전
# ============================================================
# 기능
# 1. 저장된 학습 모델 불러오기
# 2. 웹사이트에서 MP3 음원 업로드
# 3. 음원을 3초 단위로 나누어 분석
# 4. 각 구간의 MFCC 특징 추출
# 5. 7개 클래스별 예상 확률 계산
# 6. 전체 음원에 대한 평균 확률 계산
# 7. 가장 높은 확률의 종을 최종 동정
# 8. 결과를 웹페이지에 표시
#
# 실행 방법
# 1. 이 파일을 모델 파일과 같은 폴더에 저장
# 2. app.py 실행
# 3. 인터넷 브라우저에서
#    http://127.0.0.1:5000
#    접속
# ============================================================


# ------------------------------------------------------------
# 1. 필요한 프로그램 자동 설치
# ------------------------------------------------------------

import os
import sys
import subprocess

required_packages = {
    "numpy": "numpy",
    "librosa": "librosa",
    "tensorflow": "tensorflow",
    "flask": "flask"
}

for import_name, package_name in required_packages.items():

    try:
        __import__(import_name)

    except ImportError:

        print(f"{package_name} 설치 중...")

        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                package_name
            ]
        )

        print(f"{package_name} 설치 완료!")

# ------------------------------------------------------------
# 2. 라이브러리 불러오기
# ------------------------------------------------------------

from pathlib import Path
import numpy as np
import librosa
import tensorflow as tf

# ------------------------------------------------------------
# Render 무료 서버 메모리 절약 설정
# ------------------------------------------------------------

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

from flask import (
    Flask,
    request,
    render_template_string
)

# ------------------------------------------------------------
# 3. 모델 및 설정
# ------------------------------------------------------------

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
N_MFCC = 40

# ------------------------------------------------------------
# 4. Flask 앱 만들기
# ------------------------------------------------------------

app = Flask(__name__)

# 업로드 가능한 최대 파일 크기
# 30MB
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024

# ------------------------------------------------------------
# 5. 모델 저장 위치 찾기
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
model_path = BASE_DIR / "model" / "model.keras"

if not model_path.exists():
    print("모델 파일을 찾을 수 없습니다.")
    print(f"찾는 위치: {model_path}")
    raise FileNotFoundError(f"모델 파일이 없습니다: {model_path}")

# ------------------------------------------------------------
# 7. 모델 불러오기
# ------------------------------------------------------------

print()
print("=" * 60)
print("걸생 베타 매미 종 동정 모델")
print("=" * 60)

print()
print("저장된 모델을 불러오는 중...")
print(model_path)

model = tf.keras.models.load_model(
    model_path
)

print()
print("모델 불러오기 완료!")

# ------------------------------------------------------------
# 8. 웹페이지 HTML
# ------------------------------------------------------------

HTML_PAGE = """

<!DOCTYPE html>

<html lang="ko">

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>걸생 베타 매미 종 동정</title>


<style>

* {
    box-sizing: border-box;
}


body {

    margin: 0;

    font-family:
        Arial,
        "Malgun Gothic",
        sans-serif;

    background: #f1f7f2;

    color: #26352b;
}


.container {

    max-width: 700px;

    margin: 40px auto;

    padding: 20px;
}


.header {

    text-align: center;

    margin-bottom: 30px;
}


.header h1 {

    font-size: 30px;

    margin-bottom: 10px;

}


.header p {

    color: #607064;

    line-height: 1.6;
}


.card {

    background: white;

    border-radius: 20px;

    padding: 30px;

    box-shadow:
        0 5px 20px
        rgba(0, 0, 0, 0.08);

    margin-bottom: 20px;
}


.upload-area {

    border: 2px dashed #8abf91;

    border-radius: 15px;

    padding: 35px 20px;

    text-align: center;

    margin-bottom: 20px;

    background: #f8fcf8;
}


.upload-area input {

    margin-top: 15px;

    max-width: 100%;
}


.button {

    width: 100%;

    padding: 15px;

    border: none;

    border-radius: 12px;

    background: #5b9d68;

    color: white;

    font-size: 17px;

    font-weight: bold;

    cursor: pointer;
}


.button:hover {

    background: #4b8957;
}


.result {

    margin-top: 25px;
}


.final-result {

    text-align: center;

    background: #edf8ef;

    border-radius: 15px;

    padding: 25px;

    margin-bottom: 25px;
}


.final-result .species {

    font-size: 30px;

    font-weight: bold;

    margin: 10px 0;
}


.final-result .percentage {

    font-size: 22px;
}


.probability-row {

    margin-bottom: 15px;
}


.probability-name {

    display: flex;

    justify-content: space-between;

    margin-bottom: 5px;

    font-weight: bold;
}


.progress-background {

    width: 100%;

    height: 14px;

    background: #e4e9e5;

    border-radius: 10px;

    overflow: hidden;
}


.progress-bar {

    height: 100%;

    background: #6cab76;

    border-radius: 10px;
}


.info {

    font-size: 13px;

    color: #68746b;

    line-height: 1.6;

    margin-top: 25px;
}


.loading {

    text-align: center;

    padding: 20px;

    font-weight: bold;

    color: #52705a;
}


.error {

    background: #fff0f0;

    color: #a33b3b;

    padding: 15px;

    border-radius: 10px;

    margin-top: 20px;
}


.footer {

    text-align: center;

    color: #829087;

    font-size: 13px;

    margin-top: 25px;
}


</style>

</head>


<body>


<div class="container">


    <div class="header">

        <h1>
            🦗 걸생 베타 매미 종 동정
        </h1>

        <p>
            매미의 울음소리를 업로드하면<br>
            AI가 6종의 매미와 배경소리를 분석합니다.
        </p>

    </div>



    <div class="card">


        <form
            id="uploadForm"
            enctype="multipart/form-data"
        >


            <div class="upload-area">

                <strong>
                    🎵 매미 음원 선택
                </strong>

                <br>

                <input
                    type="file"
                    name="audio"
                    id="audio"
                    accept=".mp3,.wav,.m4a,.flac"
                    required
                >

                <p>
                    MP3, WAV, M4A, FLAC 지원
                </p>

            </div>


            <button
                type="submit"
                class="button"
            >
                🔍 매미 종 동정하기
            </button>


        </form>


        <div
            id="loading"
            class="loading"
            style="display:none;"
        >

            🎧 음원을 분석하고 있습니다...<br>
            잠시만 기다려 주세요.

        </div>


        <div
            id="error"
            class="error"
            style="display:none;"
        >
        </div>


        <div
            id="result"
            class="result"
            style="display:none;"
        >


            <div class="final-result">

                <div>
                    최종 동정 결과
                </div>

                <div
                    id="bestSpecies"
                    class="species"
                >
                </div>

                <div
                    id="bestPercentage"
                    class="percentage"
                >
                </div>

            </div>


            <h3>
                종별 예상 비율
            </h3>


            <div id="probabilities">
            </div>


            <div class="info">

                ※ 표시되는 확률은 실제 야외에서 해당 종일
                가능성을 의미하는 절대적인 확률이 아니라,
                AI 모델이 입력된 음원에 대해 계산한
                상대적인 예측값입니다.

            </div>


        </div>


    </div>


    <div class="card">

        <strong>
            분석 방법
        </strong>

        <p>
            입력된 음원을 3초 단위로 나눈 뒤
            각 구간의 MFCC 특징을 추출하고,
            학습된 AI 모델을 이용하여
            7개 클래스의 예측값을 계산합니다.
        </p>

    </div>


    <div class="footer">

        걸어다니는 생태도감 · 생태정보분과

    </div>


</div>



<script>


document
.getElementById("uploadForm")
.addEventListener(
    "submit",
    async function(event) {


        event.preventDefault();


        const fileInput =
            document.getElementById("audio");


        const file =
            fileInput.files[0];


        if (!file) {

            alert(
                "음원 파일을 선택해주세요."
            );

            return;
        }


        const formData =
            new FormData();


        formData.append(
            "audio",
            file
        );


        document
        .getElementById("loading")
        .style.display = "block";


        document
        .getElementById("result")
        .style.display = "none";


        document
        .getElementById("error")
        .style.display = "none";


        try {


            const response =
                await fetch(
                    "/predict",
                    {
                        method: "POST",
                        body: formData
                    }
                );


            const data =
                await response.json();


            if (!response.ok) {

                throw new Error(
                    data.error ||
                    "분석 중 오류가 발생했습니다."
                );

            }


            document
            .getElementById("loading")
            .style.display = "none";


            document
            .getElementById("result")
            .style.display = "block";


            document
            .getElementById("bestSpecies")
            .textContent =
                data.best_species;


            document
            .getElementById("bestPercentage")
            .textContent =
                data.best_percentage.toFixed(2)
                + "%";


            const probabilityArea =
                document
                .getElementById(
                    "probabilities"
                );


            probabilityArea.innerHTML = "";


            for (
                const item
                of data.probabilities
            ) {


                const row =
                    document.createElement(
                        "div"
                    );


                row.className =
                    "probability-row";


                row.innerHTML = `

                    <div class="probability-name">

                        <span>
                            ${item.species}
                        </span>

                        <span>
                            ${item.percentage.toFixed(2)}%
                        </span>

                    </div>


                    <div class="progress-background">

                        <div
                            class="progress-bar"
                            style="
                                width:
                                ${item.percentage}%;
                            "
                        >
                        </div>

                    </div>

                `;


                probabilityArea.appendChild(
                    row
                );

            }


        }

        catch (error) {


            document
            .getElementById("loading")
            .style.display = "none";


            document
            .getElementById("error")
            .style.display = "block";


            document
            .getElementById("error")
            .textContent =
                "오류: "
                + error.message;


        }


    }
);


</script>


</body>

</html>

"""


# ------------------------------------------------------------
# 9. 메인 페이지
# ------------------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    return render_template_string(
        HTML_PAGE
    )


# ------------------------------------------------------------
# 10. 음원 분석
# ------------------------------------------------------------

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    # --------------------------------------------------------
    # 음원 파일 확인
    # --------------------------------------------------------

    if "audio" not in request.files:
        return {
            "error":
                "음원 파일이 전송되지 않았습니다."
        }, 400

    file = request.files["audio"]

    if file.filename == "":
        return {
            "error":
                "음원 파일을 선택해주세요."
        }, 400

    import tempfile

    temp_path = None

    try:

        # ----------------------------------------------------
        # 파일 확장자 확인
        # ----------------------------------------------------

        original_name = file.filename

        extension = os.path.splitext(
            original_name
        )[1].lower()

        allowed_extensions = [
            ".mp3",
            ".wav",
            ".m4a",
            ".flac"
        ]

        if extension not in allowed_extensions:
            return {
                "error":
                    "MP3, WAV, M4A, FLAC 파일만 사용할 수 있습니다."
            }, 400

        # ----------------------------------------------------
        # 임시 파일 저장
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=extension
        ) as temp_file:

            file.save(
                temp_file.name
            )

            temp_path = temp_file.name

        print()
        print(
            "웹사이트 음원 분석:",
            original_name
        )

        # ----------------------------------------------------
        # 음원 불러오기
        # 최대 180초까지만 분석하여 메모리 사용 제한
        # ----------------------------------------------------

        audio, sr = librosa.load(
            temp_path,
            sr=SAMPLE_RATE,
            mono=True,
            duration=180
        )

        audio_length = (
            len(audio)
            / SAMPLE_RATE
        )

        print(
            f"분석 음원 길이: {audio_length:.2f}초"
        )

        # ----------------------------------------------------
        # 3초 단위 분석
        # 모든 구간을 리스트에 저장하지 않음
        # ----------------------------------------------------

        segment_samples = (
            SAMPLE_RATE
            * SEGMENT_SECONDS
        )

        total_segments = int(
            np.ceil(
                len(audio)
                / segment_samples
            )
        )

        print(
            f"분석 구간: {total_segments}개"
        )

        # ----------------------------------------------------
        # 확률을 누적해서 평균 계산
        # predictions 배열 전체를 저장하지 않음
        # ----------------------------------------------------

        probability_sum = np.zeros(
            len(CLASS_NAMES),
            dtype=np.float32
        )

        analyzed_count = 0

        # ----------------------------------------------------
        # 3초 구간을 하나씩 분석
        # ----------------------------------------------------

        for start in range(
                0,
                len(audio),
                segment_samples
        ):

            segment = audio[
                start:
                start + segment_samples
            ]

            # 마지막 구간이 짧으면 0으로 채움
            if len(segment) < segment_samples:
                segment = np.pad(
                    segment,
                    (
                        0,
                        segment_samples
                        - len(segment)
                    )
                )

            # ------------------------------------------------
            # MFCC 추출
            # ------------------------------------------------

            mfcc = librosa.feature.mfcc(
                y=segment,
                sr=SAMPLE_RATE,
                n_mfcc=N_MFCC
            )

            mfcc_mean = np.mean(
                mfcc,
                axis=1
            )

            # 모델 입력 형태: (1, 40)
            X_one = np.expand_dims(
                mfcc_mean,
                axis=0
            )

            # ------------------------------------------------
            # 현재 3초 구간만 예측
            # ------------------------------------------------

            prediction = model.predict(
                X_one,
                verbose=0
            )[0]

            probability_sum += prediction
            analyzed_count += 1

            # 현재 구간 데이터 즉시 삭제
            del segment
            del mfcc
            del mfcc_mean
            del X_one
            del prediction

        # ----------------------------------------------------
        # 평균 확률 계산
        # ----------------------------------------------------

        if analyzed_count == 0:
            return {
                "error":
                    "분석할 수 있는 음원이 없습니다."
            }, 400

        average_prediction = (
            probability_sum
            / analyzed_count
        )

        percentages = (
            average_prediction
            * 100
        )

        # ----------------------------------------------------
        # 가장 높은 확률의 종
        # ----------------------------------------------------

        best_index = np.argmax(
            average_prediction
        )

        best_species = (
            CLASS_NAMES[
                best_index
            ]
        )

        best_percentage = float(
            percentages[
                best_index
            ]
        )

        # ----------------------------------------------------
        # 7개 클래스 결과
        # ----------------------------------------------------

        probability_list = []

        for species, percentage in zip(
                CLASS_NAMES,
                percentages
        ):

            probability_list.append(
                {
                    "species":
                        species,

                    "percentage":
                        float(
                            percentage
                        )
                }
            )

        # ----------------------------------------------------
        # 결과 출력
        # ----------------------------------------------------

        print()
        print("=" * 60)
        print("웹사이트 동정 결과")
        print("=" * 60)

        print(
            "예상 종:",
            best_species
        )

        print(
            "예상 확률:",
            f"{best_percentage:.2f}%"
        )

        print("=" * 60)

        # ----------------------------------------------------
        # 메모리 정리
        # ----------------------------------------------------

        del audio
        del probability_sum
        del average_prediction
        del percentages

        # ----------------------------------------------------
        # 웹페이지로 JSON 결과 반환
        # ----------------------------------------------------

        return {
            "success": True,
            "filename": original_name,
            "audio_length": round(audio_length, 2),
            "segment_count": analyzed_count,
            "best_species": best_species,
            "best_percentage": best_percentage,
            "probabilities": probability_list
        }

    except Exception as e:

        print()
        print(
            "분석 오류:",
            repr(e)
        )

        return {
            "error":
                "음원 분석 중 오류가 발생했습니다.",
            "detail":
                str(e)
        }, 500

    finally:

        # ----------------------------------------------------
        # 임시 음원 삭제
        # ----------------------------------------------------

        if (
                temp_path is not None
                and os.path.exists(temp_path)
        ):

            try:
                os.remove(
                    temp_path
                )
            except Exception:
                pass


# ------------------------------------------------------------
# 11. 파일 크기 초과 오류
# ------------------------------------------------------------

@app.errorhandler(413)
def file_too_large(error):
    return {

        "error":
            "파일이 너무 큽니다. 최대 30MB까지 업로드할 수 있습니다."

    }, 413


# ------------------------------------------------------------
# 12. 프로그램 실행
# ------------------------------------------------------------

if __name__ == "__main__":
    print()
    print("=" * 60)

    print(
        "🦗 걸생 베타 매미 종 동정 웹사이트"
    )

    print("=" * 60)

    print()

    print(
        "웹사이트 주소:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print()

    print(
        "브라우저에서 위 주소를 열어주세요."
    )

    print()

    print(
        "프로그램을 종료하려면"
        " Ctrl + C 를 누르세요."
    )

    print("=" * 60)

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
