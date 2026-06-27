"""
Preprocess a raw uploaded video for inference (Member D backend).

Same lip pipeline as training data:
  raw video -> MediaPipe lip crop -> grayscale 96x96 -> (20, 96, 96) numpy array

Member D wires:  upload -> preprocess_upload_video() -> Person C predict_word()

Usage:
    from preprocess_upload import preprocess_upload_video

    clip = preprocess_upload_video("uploaded.mp4")   # shape (20, 96, 96)
    word = predict_word(clip)
"""

import cv2
import mediapipe as mp
import numpy as np

# Must match extract_lips.py / training data
LIP_WIDTH = 96
LIP_HEIGHT = 96
PADDING = 0.20
TARGET_FRAMES = 20

LIP_LANDMARKS = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308,
    324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37,
    0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81,
    42, 183, 78,
]


def lip_bbox_from_landmarks(landmarks, width, height, padding=PADDING):
    xs = [landmarks[i].x * width for i in LIP_LANDMARKS]
    ys = [landmarks[i].y * height for i in LIP_LANDMARKS]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding
    x1 = max(0, int(x1 - pad_x))
    y1 = max(0, int(y1 - pad_y))
    x2 = min(width, int(x2 + pad_x))
    y2 = min(height, int(y2 + pad_y))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def crop_and_resize(frame_gray, bbox):
    x1, y1, x2, y2 = bbox
    crop = frame_gray[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (LIP_WIDTH, LIP_HEIGHT), interpolation=cv2.INTER_AREA)


def normalize_frame_count(frames, target=TARGET_FRAMES):
    t = len(frames)
    if t == 0:
        raise ValueError("No frames to normalize")
    if t == target:
        return frames
    if t > target:
        idx = np.linspace(0, t - 1, target).astype(int)
        return [frames[i] for i in idx]
    out = list(frames)
    last = frames[-1]
    while len(out) < target:
        out.append(last.copy())
    return out


def preprocess_upload_video(video_path, start_frame=None, end_frame=None):
    """
    Process a raw uploaded video into a training-compatible numpy clip.

    Args:
        video_path: path to .mp4, .mpg, .avi, etc.
        start_frame: optional first frame (default: 0)
        end_frame: optional last frame (default: end of video)

    Returns:
        numpy array shape (20, 96, 96), dtype uint8, grayscale lip crops

    Raises:
        FileNotFoundError, RuntimeError if video cannot be read or no face found
    """
    if not video_path or not __import__("os").path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = 0 if start_frame is None else max(0, int(start_frame))
    end = (total - 1) if end_frame is None else min(int(end_frame), total - 1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    frames = []
    last_bbox = None

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        for _ in range(start, end + 1):
            ok, frame = cap.read()
            if not ok:
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)

            bbox = None
            if result.multi_face_landmarks:
                bbox = lip_bbox_from_landmarks(
                    result.multi_face_landmarks[0].landmark, w, h
                )
            if bbox is None and last_bbox is not None:
                bbox = last_bbox
            if bbox is None:
                continue

            last_bbox = bbox
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            lip = crop_and_resize(gray, bbox)
            if lip is not None:
                frames.append(lip)

    cap.release()

    if not frames:
        raise RuntimeError("No lip frames extracted — face not detected in video")

    frames = normalize_frame_count(frames)
    return np.stack(frames, axis=0).astype(np.uint8)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python preprocess_upload.py <video_path>")
        sys.exit(1)
    arr = preprocess_upload_video(sys.argv[1])
    print("OK shape:", arr.shape, "dtype:", arr.dtype)
    