from unittest import result

import numpy as np
import sys
import os
from keras import backend as K
from keras.optimizers import Adam
from lipnet.model2 import LipNet
from lipnet.lipreading.videos import Video
from lipnet.core.decoders import Decoder
from lipnet.lipreading.helpers import labels_to_text
from lipnet.utils.spell import Spell
from preprocess import extract

# LipNet expects frames of this shape
IMG_C, IMG_W, IMG_H = 3, 100, 50
FRAMES_N = 75
ABSOLUTE_MAX_STRING_LEN = 32

def predict_from_video(video_path):
    print(f"Processing: {video_path}")

    temp_path = "temp_frames.npy"

    print("STEP 1")
    frames = extract(video_path, temp_path)

    print("STEP 2")

    if len(frames) == 0:
        print("No frames extracted")
        return ""

    print("Frames extracted:", len(frames))

    lipnet = LipNet(
        img_c=IMG_C,
        img_w=IMG_W,
        img_h=IMG_H,
        frames_n=FRAMES_N,
        absolute_max_string_len=ABSOLUTE_MAX_STRING_LEN,
        output_size=28
    )

    print("STEP 3")

    adam = Adam(learning_rate=0.0001)

    lipnet.model.compile(
        loss={'ctc': lambda y_true, y_pred: y_pred},
        optimizer=adam
    )

    print("STEP 4")

   # Load pretrained weights
    weights_path = os.path.join("evaluation","models","overlapped-weights368.h5")

    if not os.path.exists(weights_path):
        print(f"Weights file not found at {weights_path}")
        return ""

    print("Loading weights from:", weights_path)

    lipnet.model.load_weights(weights_path)

    print("STEP 5")

# frames already returned from extract()
# shape should be: (75, 50, 100)

    frames = frames.astype(np.float32) / 255.0

# convert grayscale -> 3 channels
    frames = np.stack([frames, frames, frames], axis=-1)

    print("STEP 6", frames.shape)

# LipNet expects: (frames, width, height, channels)
    frames = np.transpose(frames, (0, 2, 1, 3))

    print("STEP 7", frames.shape)

# ensure exactly 75 frames
    if len(frames) < FRAMES_N:

        pad = np.zeros(
            (FRAMES_N - len(frames), IMG_W, IMG_H, IMG_C),
            dtype=np.float32
        )

        frames = np.concatenate([frames, pad], axis=0)

    else:
        frames = frames[:FRAMES_N]

    print("STEP 8", frames.shape)

# add batch dimension
    frames = np.expand_dims(frames, axis=0)

    print("STEP 9", frames.shape)

# predict
    y_pred = lipnet.predict(frames)

    print("STEP 10")

# decode prediction
    input_length = np.array([FRAMES_N])

    decoder = Decoder(
        greedy=True,
        beam_width=200,
        postprocessors=[labels_to_text])

    result = decoder.decode(y_pred, input_length)

    print("Prediction:", result)

    return result
if __name__ == "__main__":
    result = predict_from_video("data/s1/videos/bbaf2n.mpg")
    print("FINAL RESULT:", result)