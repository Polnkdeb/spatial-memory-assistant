import json
import os
import cv2


class Searcher:
    def __init__(self, index_path, frames_dir):
        with open(index_path, "r") as f:
            self.index = json.load(f)

        self.frames_dir = frames_dir

    def search(self, query_text, top_k=5):
        query_lower = query_text.lower()

        if query_lower not in self.index:
            print(f"Объект '{query_text}' не найден в индексе.")
            return []

        results = self.index[query_lower]
        return results[:top_k]

    def show_results(self, query_text, top_k=5):
        results = self.search(query_text, top_k)

        if not results:
            return

        print(f"Найдено {len(results)} кадров для '{query_text}':")

        for result in results:
            frame_file = result["frame"]
            conf = result["confidence"]
            bbox = result["bbox"]

            frame_path = os.path.join(self.frames_dir, frame_file)
            frame = cv2.imread(frame_path)

            if frame is None:
                print(f"Не удалось открыть кадр: {frame_path}")
                continue

            x1, y1, x2, y2 = bbox

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{query_text} {conf:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

            cv2.imshow("Result", frame)
            print(f"  - {frame_file} (conf={conf:.2f})")
            cv2.waitKey(0)

        cv2.destroyAllWindows()