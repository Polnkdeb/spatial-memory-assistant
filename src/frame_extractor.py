import cv2
import os

def extract_frames(video_path, output_dir):
    """
    Извлекает кадры из видео (один кадр в секунду) и сохраняет в output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Ошибка: не удалось открыть видео {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(int(fps), 1)  
    frame_count = 0
    saved_count = 0

    print(f"Извлечение кадров из {video_path} (FPS={fps}, интервал={frame_interval})")
    while True:
        success, frame = cap.read()
        if not success:
            break
        if frame_count % frame_interval == 0:
            filename = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1
        frame_count += 1

    cap.release()
    print(f"Готово. Сохранено кадров: {saved_count} из {frame_count}")