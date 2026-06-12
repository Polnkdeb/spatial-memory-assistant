import os
import json
from src.frame_extractor import extract_frames
from src.detector import YOLODetector
from src.indexer import IndexBuilder
from src.query import Searcher
from src.coco_classes import get_available_classes

VIDEO_PATH = "data/videos/IMG_2992.MOV"
FRAMES_DIR = "data/frames"
DETECTIONS_JSON = "data/detections.json"
INDEX_JSON = "data/index.json"

def main():
    print("=== Spatial Memory Assistant ===\n")

    if not os.path.exists(FRAMES_DIR) or len(os.listdir(FRAMES_DIR)) == 0:
        print("Шаг 1: Извлечение кадров...")
        extract_frames(VIDEO_PATH, FRAMES_DIR)
    else:
        print("Шаг 1 пропущен: кадры уже есть.")

    if not os.path.exists(DETECTIONS_JSON):
        print("\nШаг 2: Детекция объектов...")
        detector = YOLODetector()
        detector.detect_on_frames(FRAMES_DIR, DETECTIONS_JSON)
    else:
        print("Шаг 2 пропущен: detections.json уже существует.")

    if not os.path.exists(INDEX_JSON):
        print("\nШаг 3: Построение индекса...")
        IndexBuilder.build_from_detections(DETECTIONS_JSON, INDEX_JSON)
    else:
        print("Шаг 3 пропущен: index.json уже существует.")

    print("\nШаг 4: Поиск объектов")
    searcher = Searcher(INDEX_JSON, FRAMES_DIR)

    # Использую статический список классов COCO
    all_classes = get_available_classes()
    print(f"Доступно 80 классов (YOLO COCO). Первые 20: {', '.join(all_classes[:20])}")
    print("Полный список: src/coco_classes.py\n")

    while True:
        query = input("Что ищем? (на английском, или 'выход'): ").strip().lower()
        if query == "выход":
            break
        searcher.show_results(query, top_k=5)

if __name__ == "__main__":
    main()