# ============================================================
# 걸생 베타 매미 종 동정 모델 3
# Mel-Spectrogram + CNN
# ============================================================

import sys
import subprocess
import os
import random


# ============================================================
# 1. 필요한 프로그램 자동 설치
# ============================================================

packages = [
    "numpy",
    "librosa",
    "scikit-learn",
    "tensorflow",
    "streamlit",
    "soundfile"
]

print("=" * 60)
print("필요한 프로그램을 확인합니다.")
print("=" * 60)

for package in packages:

    try:

        if package == "scikit-learn":
            import sklearn

        else:
            __import__(package)

        print("✓", package, "설치됨")

    except ImportError:

        print("→", package, "설치 중...")

        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            package
        ])


print()
print("모든 프로그램 설치 완료!")


# ============================================================
# 2. 라이브러리
# ============================================================

import numpy as np
import librosa
import tensorflow as tf

from sklearn.model_selection import train_test_split


# ============================================================
# 3. 기본 설정
# ============================================================

MODEL_NAME = "걸생 베타 매미 종 동정 모델"

SPECIES = [
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

EPOCHS = 80

BATCH_SIZE = 8

RANDOM_SEED = 42


np.random.seed(RANDOM_SEED)

random.seed(RANDOM_SEED)

tf.random.set_seed(RANDOM_SEED)


# ============================================================
# 4. 바탕화면 찾기
# ============================================================

def find_desktop():

    home = os.path.expanduser("~")

    possible_paths = [

        os.path.join(
            home,
            "Desktop"
        ),

        os.path.join(
            home,
            "OneDrive",
            "Desktop"
        ),

        os.path.join(
            home,
            "바탕 화면"
        ),

        os.path.join(
            home,
            "OneDrive",
            "바탕 화면"
        )
    ]

    for path in possible_paths:

        if os.path.exists(path):

            model_folder = os.path.join(
                path,
                MODEL_NAME
            )

            if os.path.exists(model_folder):

                return path

    return None


DESKTOP = find_desktop()


if DESKTOP is None:

    print()
    print("❌ '걸생 베타 매미 종 동정 모델' 폴더를 찾지 못했습니다.")

    print()
    print("바탕화면에 다음 폴더가 있는지 확인하세요.")

    print()
    print("걸생 베타 매미 종 동정 모델")

    input(
        "\nEnter를 누르면 종료합니다."
    )

    sys.exit()


# ============================================================
# 5. 폴더
# ============================================================

BASE_FOLDER = os.path.join(

    DESKTOP,

    MODEL_NAME

)


DATA_FOLDER = os.path.join(

    BASE_FOLDER,

    "울음소리 표본"

)


print()
print("=" * 60)

print("모델 폴더:")
print(BASE_FOLDER)

print()

print("음원 폴더:")
print(DATA_FOLDER)

print("=" * 60)


if not os.path.exists(DATA_FOLDER):

    print()
    print("❌ 울음소리 표본 폴더가 없습니다.")

    input(
        "\nEnter를 누르면 종료합니다."
    )

    sys.exit()


# ============================================================
# 6. 음원 파일 찾기
# ============================================================

print()
print("=" * 60)
print("음원 파일 확인")
print("=" * 60)


all_files = []


for species_index, species in enumerate(
    SPECIES
):

    species_folder = os.path.join(

        DATA_FOLDER,

        species

    )


    if not os.path.exists(
        species_folder
    ):

        print(
            "❌ 폴더 없음:",
            species
        )

        continue


    files = []


    for file_name in os.listdir(
        species_folder
    ):

        if file_name.lower().endswith(
            (
                ".mp3",
                ".wav",
                ".m4a",
                ".flac"
            )
        ):

            files.append(

                os.path.join(

                    species_folder,

                    file_name

                )

            )


    print(
        species,
        ":",
        len(files),
        "개"
    )


    for file_path in files:

        all_files.append(

            (
                file_path,
                species_index
            )

        )


# ============================================================
# 7. 음원 존재 확인
# ============================================================

if len(all_files) == 0:

    print()
    print("❌ 음원 파일을 하나도 찾지 못했습니다.")

    input(
        "\nEnter를 누르면 종료합니다."
    )

    sys.exit()


print()
print(
    "전체 원본 음원:",
    len(all_files),
    "개"
)


# ============================================================
# 8. 원본 파일 기준 학습 / 검증 분리
# ============================================================

file_paths = np.array(

    [
        item[0]
        for item in all_files
    ]

)


file_labels = np.array(

    [
        item[1]
        for item in all_files
    ]

)


print()
print("=" * 60)
print("학습 / 검증 파일 분리")
print("=" * 60)


X_train_files, X_test_files, y_train_files, y_test_files = (

    train_test_split(

        file_paths,

        file_labels,

        test_size=0.33,

        random_state=RANDOM_SEED,

        stratify=file_labels

    )

)


print()
print(
    "학습용 원본:",
    len(X_train_files),
    "개"
)


print(
    "검증용 원본:",
    len(X_test_files),
    "개"
)


# ============================================================
# 9. 음원 불러오기
# ============================================================

def load_audio(file_path):

    audio, sr = librosa.load(

        file_path,

        sr=SAMPLE_RATE,

        mono=True

    )

    return audio


# ============================================================
# 10. 3초 단위 분할
# ============================================================

def split_audio(audio):

    segment_length = (

        SAMPLE_RATE *
        SEGMENT_SECONDS

    )


    segments = []


    # 3초보다 짧은 경우

    if len(audio) < segment_length:

        audio = np.pad(

            audio,

            (
                0,

                segment_length - len(audio)

            )

        )

        segments.append(
            audio
        )

        return segments


    # 3초씩 자르기

    for start in range(

        0,

        len(audio) - segment_length + 1,

        segment_length

    ):

        segment = audio[

            start:
            start + segment_length

        ]


        segments.append(
            segment
        )


    return segments


# ============================================================
# 11. 음원 증강
# ============================================================

def augment_audio(audio):

    method = random.choice(

        [
            "volume",
            "noise",
            "shift"
        ]

    )


    # --------------------------------------------------------
    # 음량 변화
    # --------------------------------------------------------

    if method == "volume":

        gain = random.uniform(

            0.7,

            1.3

        )

        audio = audio * gain


    # --------------------------------------------------------
    # 작은 잡음
    # --------------------------------------------------------

    elif method == "noise":

        noise_level = random.uniform(

            0.002,

            0.01

        )


        noise = np.random.normal(

            0,

            noise_level,

            len(audio)

        )


        audio = audio + noise


    # --------------------------------------------------------
    # 시간 이동
    # --------------------------------------------------------

    elif method == "shift":

        max_shift = int(

            SAMPLE_RATE * 0.2

        )


        shift = random.randint(

            -max_shift,

            max_shift

        )


        audio = np.roll(

            audio,

            shift

        )


    # --------------------------------------------------------
    # 음량 제한
    # --------------------------------------------------------

    maximum = np.max(

        np.abs(audio)

    )


    if maximum > 1:

        audio = (

            audio /
            maximum

        )


    return audio.astype(
        np.float32
    )


# ============================================================
# 12. Mel-Spectrogram
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


    # 정규화

    mel_db = (

        mel_db + 80

    ) / 80


    return mel_db.astype(
        np.float32
    )


# ============================================================
# 13. 학습 데이터
# ============================================================

X_train = []

y_train = []


print()
print("=" * 60)
print("학습 데이터 생성")
print("=" * 60)


for file_path, label in zip(

    X_train_files,

    y_train_files

):

    print(

        "학습:",
        SPECIES[label],
        "|",
        os.path.basename(file_path)

    )


    try:

        audio = load_audio(
            file_path
        )


        segments = split_audio(
            audio
        )


        for segment in segments:

            # ------------------------------------------------
            # 원본
            # ------------------------------------------------

            mel = make_mel_spectrogram(

                segment

            )


            X_train.append(
                mel
            )


            y_train.append(
                label
            )


            # ------------------------------------------------
            # 증강 데이터 2개
            # ------------------------------------------------

            for _ in range(2):

                augmented = augment_audio(

                    segment

                )


                mel = make_mel_spectrogram(

                    augmented

                )


                X_train.append(
                    mel
                )


                y_train.append(
                    label
                )


    except Exception as e:

        print(
            "⚠ 파일 처리 오류:",
            e
        )


# ============================================================
# 14. 검증 데이터
# ============================================================

X_test = []

y_test = []


print()
print("=" * 60)
print("검증 데이터 생성")
print("=" * 60)


for file_path, label in zip(

    X_test_files,

    y_test_files

):

    print(

        "검증:",
        SPECIES[label],
        "|",
        os.path.basename(file_path)

    )


    try:

        audio = load_audio(
            file_path
        )


        segments = split_audio(
            audio
        )


        for segment in segments:

            # 검증 데이터는 증강하지 않음

            mel = make_mel_spectrogram(

                segment

            )


            X_test.append(
                mel
            )


            y_test.append(
                label
            )


    except Exception as e:

        print(
            "⚠ 파일 처리 오류:",
            e
        )


# ============================================================
# 15. NumPy 변환
# ============================================================

X_train = np.array(

    X_train,

    dtype=np.float32

)


X_test = np.array(

    X_test,

    dtype=np.float32

)


y_train = np.array(

    y_train,

    dtype=np.int32

)


y_test = np.array(

    y_test,

    dtype=np.int32

)


# ============================================================
# 16. CNN 입력 형태
# ============================================================

X_train = X_train[
    ..., np.newaxis
]


X_test = X_test[
    ..., np.newaxis
]


print()
print("=" * 60)
print("데이터 준비 완료")
print("=" * 60)


print()
print(
    "학습 데이터:",
    X_train.shape
)


print(
    "검증 데이터:",
    X_test.shape
)


# ============================================================
# 17. CNN 모델
# ============================================================

model = tf.keras.Sequential([

    tf.keras.layers.Input(

        shape=X_train.shape[1:]

    ),


    # --------------------------------------------------------
    # CNN 1
    # --------------------------------------------------------

    tf.keras.layers.Conv2D(

        32,

        (3, 3),

        padding="same",

        activation="relu"

    ),


    tf.keras.layers.BatchNormalization(),


    tf.keras.layers.MaxPooling2D(

        (2, 2)

    ),


    tf.keras.layers.Dropout(

        0.20

    ),


    # --------------------------------------------------------
    # CNN 2
    # --------------------------------------------------------

    tf.keras.layers.Conv2D(

        64,

        (3, 3),

        padding="same",

        activation="relu"

    ),


    tf.keras.layers.BatchNormalization(),


    tf.keras.layers.MaxPooling2D(

        (2, 2)

    ),


    tf.keras.layers.Dropout(

        0.25

    ),


    # --------------------------------------------------------
    # CNN 3
    # --------------------------------------------------------

    tf.keras.layers.Conv2D(

        128,

        (3, 3),

        padding="same",

        activation="relu"

    ),


    tf.keras.layers.BatchNormalization(),


    tf.keras.layers.MaxPooling2D(

        (2, 2)

    ),


    tf.keras.layers.Dropout(

        0.30

    ),


    # --------------------------------------------------------
    # 특징 압축
    # --------------------------------------------------------

    tf.keras.layers.GlobalAveragePooling2D(),


    # --------------------------------------------------------
    # 분류
    # --------------------------------------------------------

    tf.keras.layers.Dense(

        128,

        activation="relu"

    ),


    tf.keras.layers.Dropout(

        0.40

    ),


    tf.keras.layers.Dense(

        len(SPECIES),

        activation="softmax"

    )

])


# ============================================================
# 18. 학습 설정
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(

        learning_rate=0.0005

    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)


# ============================================================
# 19. 조기 종료
# ============================================================

early_stopping = tf.keras.callbacks.EarlyStopping(

    monitor="val_loss",

    patience=12,

    restore_best_weights=True

)


# ============================================================
# 20. 학습 시작
# ============================================================

print()
print("=" * 60)
print("매미모델3 학습 시작")
print("=" * 60)
print()


history = model.fit(

    X_train,

    y_train,

    validation_data=(

        X_test,

        y_test

    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=[

        early_stopping

    ],

    verbose=1

)


# ============================================================
# 21. 평가
# ============================================================

loss, accuracy = model.evaluate(

    X_test,

    y_test,

    verbose=0

)


print()
print("=" * 60)
print("학습 완료")
print("=" * 60)


print()
print(
    "검증 정확도:",
    round(

        accuracy * 100,

        2

    ),
    "%"
)


# ============================================================
# 22. 모델 저장
# ============================================================

model_path = os.path.join(

    BASE_FOLDER,

    "매미모델3.keras"

)


model.save(

    model_path

)


print()
print("=" * 60)
print("모델 저장 완료!")
print("=" * 60)


print()
print("저장 위치:")
print(model_path)


# ============================================================
# 23. 클래스 정보 저장
# ============================================================

class_path = os.path.join(

    BASE_FOLDER,

    "매미클래스3.txt"

)


with open(

    class_path,

    "w",

    encoding="utf-8"

) as f:

    for i, species in enumerate(

        SPECIES

    ):

        f.write(

            f"{i}: {species}\n"

        )


print()
print("클래스 정보:")
print(class_path)


# ============================================================
# 24. 최종 결과
# ============================================================

print()
print("=" * 60)
print("「매미모델3」 제작 완료!")
print("=" * 60)


print()
print("모델 파일:")

print(
    "매미모델3.keras"
)


print()
print("클래스:")

for i, species in enumerate(
    SPECIES
):

    print(
        i,
        "→",
        species
    )


print()
print("이제 이 모델을 웹사이트에 연결할 수 있습니다.")


input(
    "\nEnter를 누르면 종료합니다."
)
