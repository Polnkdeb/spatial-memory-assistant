import json
import shutil
from pathlib import Path

import cv2
import streamlit as st

from src.frame_extractor import extract_frames
from src.detector import YOLODetector
from src.indexer import IndexBuilder


st.set_page_config(
    page_title="Spatial Memory Assistant",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Spatial Memory Assistant")
st.write("Поиск объектов на видео с помощью YOLOv8.")


WORK_DIR = Path("streamlit_data")
VIDEO_PATH = WORK_DIR / "uploaded_video.mov"
FRAMES_DIR = WORK_DIR / "frames"
DETECTIONS_JSON = WORK_DIR / "detections.json"
INDEX_JSON = WORK_DIR / "index.json"


uploaded_video = st.file_uploader(
    "Загрузите видео",
    type=["mp4", "mov", "MOV", "avi"]
)


if uploaded_video:
    if st.button("Анализировать видео"):
        # очищаем старые результаты
        if WORK_DIR.exists():
            shutil.rmtree(WORK_DIR)

        WORK_DIR.mkdir(parents=True, exist_ok=True)
        FRAMES_DIR.mkdir(parents=True, exist_ok=True)

        VIDEO_PATH.write_bytes(uploaded_video.read())

        st.info("Извлечение кадров...")
        extract_frames(str(VIDEO_PATH), str(FRAMES_DIR))

        st.info("Детекция объектов...")
        detector = YOLODetector()
        detector.detect_on_frames(str(FRAMES_DIR), str(DETECTIONS_JSON))

        st.info("Построение индекса...")
        IndexBuilder.build_from_detections(
            str(DETECTIONS_JSON),
            str(INDEX_JSON)
        )

        with open(INDEX_JSON, "r") as f:
            index = json.load(f)

        st.session_state["index"] = index

        st.success("Готово!")


if "index" in st.session_state:
    index = st.session_state["index"]

    st.subheader("Найденные объекты")

    object_names = sorted(index.keys())
    st.write(", ".join(object_names))

    query = st.selectbox(
        "Выберите объект для поиска",
        object_names
    )

    top_k = st.slider(
        "Сколько кадров показать",
        min_value=1,
        max_value=10,
        value=5
    )

    results = index[query][:top_k]

    st.subheader(f"Результаты для: {query}")

    cols = st.columns(2)

    for i, result in enumerate(results):
        frame_file = result["frame"]
        conf = result["confidence"]
        bbox = result["bbox"]

        frame_path = FRAMES_DIR / frame_file
        frame = cv2.imread(str(frame_path))

        if frame is None:
            st.warning(f"Не удалось открыть кадр: {frame_file}")
            continue

        x1, y1, x2, y2 = bbox

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            frame,
            f"{query} {conf:.2f}",
            (x1, max(y1 - 5, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with cols[i % 2]:
            st.image(
                frame_rgb,
                caption=f"{frame_file} | confidence={conf:.2f}",
                use_container_width=True
            )