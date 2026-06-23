import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset


class LipReadingDataset(Dataset):

    def __init__(self,
                 data_dir="data/train",
                 labels_file="labels.json"):

        with open(labels_file, "r") as f:
            meta = json.load(f)

        self.word_to_idx = meta["word_to_idx"]

        self.samples = []

        for word in os.listdir(data_dir):

            word_folder = os.path.join(data_dir, word)

            if not os.path.isdir(word_folder):
                continue

            label = self.word_to_idx[word]

            for file in os.listdir(word_folder):

                if file.endswith(".npy"):

                    self.samples.append(
                        (
                            os.path.join(word_folder, file),
                            label
                        )
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        file_path, label = self.samples[idx]

        clip = np.load(file_path)

        TARGET_FRAMES = 10

        num_frames = clip.shape[0]

        if num_frames < TARGET_FRAMES:

            pad_amount = TARGET_FRAMES - num_frames

            clip = np.pad(
                clip,
                ((0, pad_amount), (0, 0), (0, 0)),
                mode="constant"
            )

        elif num_frames > TARGET_FRAMES:

            clip = clip[:TARGET_FRAMES]

        clip = torch.tensor(
            clip,
            dtype=torch.float32
        )

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return clip, label