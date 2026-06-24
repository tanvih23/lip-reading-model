import cv2
import numpy as np
import os


def convert(video_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f'Video not found: {video_path}')
    return video_path


_FACE_CASCADE = cv2.CascadeClassifier(
    os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
)


def _crop_mouth(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
        x1 = max(0, x + int(w * 0.2))
        x2 = min(frame.shape[1], x + int(w * 0.8))
        y1 = max(0, y + int(h * 0.55))
        y2 = min(frame.shape[0], y + int(h * 0.95))
        mouth = frame[y1:y2, x1:x2]
        if mouth.size > 0:
            return mouth

    height, width = frame.shape[:2]
    x1 = width // 4
    x2 = width * 3 // 4
    y1 = height // 2
    y2 = min(height, y1 + height // 3)
    return frame[y1:y2, x1:x2]


def extract(video_path, save_path=None):
    video_path = convert(video_path)
    cap = cv2.VideoCapture(video_path)

    frames = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        mouth = _crop_mouth(frame)

        if mouth.size == 0:
            continue

        mouth = cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY)
        mouth = cv2.resize(mouth, (100, 50))
        frames.append(mouth)

    cap.release()

    frames = np.array(frames)

    if len(frames) > 75:
        frames = frames[:75]

    if save_path:
        np.save(save_path, frames)

    print("Frames shape:", frames.shape)

    return frames

    
def save_as_video(npy_path, output_path, fps=25):
    frames = np.load(npy_path)
    height, width = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)
    for frame in frames:
        out.write(frame)
    out.release()
    print(f"Video saved to {output_path}")


if __name__ == "__main__":
    extract(
        "data/s1/videos/bbaf2n.mpg",
        "data/processed/bbaf2n.npy"
    )
    save_as_video(
        "data/processed/bbaf2n.npy",
        "output_cropped_lips.mp4"
    )


