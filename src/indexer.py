import json
from collections import defaultdict


class IndexBuilder:
    @staticmethod
    def build_from_detections(detections_json_path, output_index_path):
        with open(detections_json_path, "r") as f:
            detections = json.load(f)

        index = defaultdict(list)

        for frame_file, objects in detections.items():
            for obj in objects:
                class_name = obj["class"]

                index[class_name].append({
                    "frame": frame_file,
                    "confidence": obj["confidence"],
                    "bbox": obj["bbox"]
                })

        for class_name in index:
            index[class_name].sort(
                key=lambda item: item["confidence"],
                reverse=True
            )

        with open(output_index_path, "w") as f:
            json.dump(index, f, indent=2)

        return index