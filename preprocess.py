import mediapipe as mp
import cv2
import numpy as np
import os
def convert(video_path):
    if video_path.endswith(".mpg") or video_path.endswith(".mpeg"):
        return video_path
    output_path = video_path.rsplit('.', 1)[0] + '_converted.mp4'
    os.system(f'ffmpeg -i "{video_path}" -vcodec libx264 "{output_path}" -y')
    return output_path
mpface_mesh=mp.solutions.face_mesh
face_mesh=mpface_mesh.FaceMesh()
LIPS=[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
                 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
def extract(video_path, save_path=None):
    video_path = convert(video_path)
    cap = cv2.VideoCapture(video_path)

    frames = []

    with mpface_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True
    ) as face_mesh:

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = face_mesh.process(frame_rgb)

            if not results.multi_face_landmarks:
                continue

            landmarks = results.multi_face_landmarks[0]

            h, w, _ = frame.shape

            lip_points = []

            for idx in LIPS:
                point = landmarks.landmark[idx]
                x = int(point.x * w)
                y = int(point.y * h)
                lip_points.append((x, y))

            lip_points = np.array(lip_points)

            x_min = np.min(lip_points[:, 0])
            x_max = np.max(lip_points[:, 0])
            y_min = np.min(lip_points[:, 1])
            y_max = np.max(lip_points[:, 1])

            pad = 20

            x_min = max(0, x_min - pad)
            y_min = max(0, y_min - pad)
            x_max = min(w, x_max + pad)
            y_max = min(h, y_max + pad)

            mouth = frame[y_min:y_max, x_min:x_max]

            if mouth.size == 0:
                continue

            mouth = cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY)

            # LipNet expects 100x50
            mouth = cv2.resize(mouth, (100, 50))

            frames.append(mouth)

    cap.release()

    frames = np.array(frames)

    # Limit to 75 frames
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


