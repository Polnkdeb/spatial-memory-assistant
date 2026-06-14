from pathlib import Path

from src.frame_extractor import extract_frames
from src.detector import YOLODetector
from src.indexer import IndexBuilder
from src.query import Searcher
from src.coco_classes import get_available_classes


VIDEO_PATH = Path("data/videos/IMG_2992.MOV")
FRAMES_DIR = Path("data/frames")
DETECTIONS_JSON = Path("data/detections.json")
INDEX_JSON = Path("data/index.json")


def main():
    print("=== Spatial Memory Assistant ===\n")

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    if not FRAMES_DIR.exists() or not any(FRAMES_DIR.iterdir()):
        print("Шаг 1: Извлечение кадров...")
        extract_frames(str(VIDEO_PATH), str(FRAMES_DIR))
    else:
        print("Шаг 1 пропущен: кадры уже есть.")

    if not DETECTIONS_JSON.exists():
        print("\nШаг 2: Детекция объектов...")
        detector = YOLODetector()
        detector.detect_on_frames(str(FRAMES_DIR), str(DETECTIONS_JSON))
    else:
        print("Шаг 2 пропущен: detections.json уже существует.")

    if not INDEX_JSON.exists():
        print("\nШаг 3: Построение индекса...")
        IndexBuilder.build_from_detections(str(DETECTIONS_JSON), str(INDEX_JSON))
    else:
        print("Шаг 3 пропущен: index.json уже существует.")

    print("\nШаг 4: Поиск объектов")
    searcher = Searcher(str(INDEX_JSON), str(FRAMES_DIR))

    all_classes = get_available_classes()
    print(f"Доступно {len(all_classes)} классов YOLO COCO.")
    print(f"Первые 20: {', '.join(all_classes[:20])}")
    print("Полный список: src/coco_classes.py\n")

    while True:
        query = input("Что ищем? (на английском, или 'выход'): ").strip().lower()

        if query in {"выход", "exit", "quit"}:
            break

        if query not in all_classes:
            print(f"Класс '{query}' не найден. Попробуй, например: person, chair, bottle.")
            continue

        searcher.show_results(query, top_k=5)


if __name__ == "__main__":
    main()