"""
inference.py

THIS FILE'S JOB:
Load the trained model checkpoint and take a single video clip as input,
returning the predicted word and confidence scores.

This is the file FastAPI backend will import and call directly.
The API will receive a video file, preprocess it, call predict_word(), and
return the result as JSON.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os

from model import LipReaderModel
from dummy_data import NUM_CLASSES, NUM_FRAMES, NUM_CHANNELS, IMG_HEIGHT, IMG_WIDTH

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

CHECKPOINT_PATH = "from_scratch/checkpoints/best_model.pth"
LABELS_PATH = "from_scratch/labels.json"

# ---------------------------------------------------------------------------
# LOAD LABELS
# ---------------------------------------------------------------------------
# We need idx_to_word to convert the model's output (a number like 3) back
# into a human-readable word (like "green"). Person B's labels.json has this.

with open(LABELS_PATH, "r") as f:
    meta = json.load(f)

# idx_to_word maps "0" -> "again", "1" -> "bin", etc.
# Keys are strings in JSON, so we convert to ints for easy lookup.
IDX_TO_WORD = {int(k): v for k, v in meta["idx_to_word"].items()}

# ---------------------------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------------------------

def load_model(checkpoint_path=CHECKPOINT_PATH):
    """
    Loads the trained model from a checkpoint file.
    Called once when the FastAPI server starts -- not on every request.

    Returns the model in eval mode, ready to make predictions.
    """

    # Detect device -- GPU if available, otherwise CPU.
    # On the deployed server this will likely be CPU, which is fine for
    # inference (single clip at a time, not batches of 720).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build the model architecture -- same structure as during training.
    # Right now the weights are random, just like Day 1.
    model = LipReaderModel(num_classes=NUM_CLASSES)

    # Load the checkpoint dictionary we saved during training.
    # map_location=device handles the case where the model was trained on
    # GPU (Colab) but is now being loaded on CPU (local server) -- without
    # this argument PyTorch would crash trying to load GPU tensors onto CPU.
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load ONLY the weights from the checkpoint into the model.
    # checkpoint["model_state_dict"] is the dictionary of all learned weights
    # we saved with torch.save() in train.py. This is the line that transforms
    # the random-weight model above into the trained model.
    model.load_state_dict(checkpoint["model_state_dict"])

    # model.eval() switches the model to inference mode.
    # This is the opposite of model.train() from train.py.
    # It disables things like Dropout (if we had it) that behave differently
    # during training vs inference, ensuring consistent predictions.
    model.eval()

    # Move model to device
    model = model.to(device)

    print(f"Model loaded from {checkpoint_path} (epoch {checkpoint['epoch']}, "
          f"loss {checkpoint['loss']:.4f})")

    return model, device


# Load the model once when this module is imported.
# This means FastAPI loads the model when the server starts, not on every
# single request -- loading a model takes a second or two, you don't want
# that delay on every API call.
model, device = load_model()


# ---------------------------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------------------------

def preprocess_clip(clip_array):
    """
    Takes a raw numpy array and prepares
    it for the model.

    Input:  numpy array, any of these shapes:
            (frames, H, W)        -- no channel dimension (Person A's format)
            (frames, 1, H, W)     -- already has channel dimension
    Output: torch tensor of shape (1, NUM_FRAMES, 1, H, W)
            -- the leading 1 is the batch dimension (batch size of 1 for
               inference, since we predict one clip at a time)
    """

    # --- Handle channel dimension ---
    if clip_array.ndim == 3:
        # Shape is (frames, H, W) -- add channel dimension
        # Same fix as Person B's dataset.py
        clip_array = clip_array[:, np.newaxis, :, :]
    # Now shape is (frames, 1, H, W)

    # --- Handle frame count ---
    num_frames = clip_array.shape[0]
    if num_frames < NUM_FRAMES:
        # Pad with zeros if clip is shorter than expected
        pad = np.zeros((NUM_FRAMES - num_frames, 1, IMG_HEIGHT, IMG_WIDTH),
                       dtype=clip_array.dtype)
        clip_array = np.concatenate([clip_array, pad], axis=0)
    elif num_frames > NUM_FRAMES:
        # Trim if longer than expected
        clip_array = clip_array[:NUM_FRAMES]

    # --- Normalize ---
    # Convert uint8 (0-255) to float32 (0.0-1.0)
    # Same normalization as dataset.py
    clip_array = clip_array.astype(np.float32) / 255.0

    # --- Convert to tensor ---
    clip_tensor = torch.tensor(clip_array)

    # --- Add batch dimension ---
    # Model expects (batch_size, frames, channels, H, W)
    # We only have one clip, so batch_size = 1
    # .unsqueeze(0) adds a new dimension at position 0
    clip_tensor = clip_tensor.unsqueeze(0)
    # Shape is now (1, NUM_FRAMES, 1, H, W) -- ready for the model

    return clip_tensor


# ---------------------------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------------------------

def predict_word(clip_array):
    """
    Takes a single lip-crop video clip as a numpy array and returns
    the predicted word with confidence scores for all 12 words.

    This is the function whic will be called from FastAPI like:
        result = predict_word(clip_array)
        return result  # send as JSON response

    Input:  numpy array of shape (frames, H, W) or (frames, 1, H, W)
    Output: dictionary with:
        {
            "predicted_word": "bin",
            "confidence": 0.94,
            "all_scores": {
                "again": 0.01,
                "bin": 0.94,
                "blue": 0.02,
                ...
            }
        }
    """

    # Preprocess the clip into the right tensor shape
    clip_tensor = preprocess_clip(clip_array)

    # Move to same device as model
    clip_tensor = clip_tensor.to(device)

    # torch.no_grad() tells PyTorch: don't track gradients during this
    # forward pass. During training we needed gradients for backprop.
    # During inference we don't -- skipping gradient tracking makes
    # inference faster and uses less memory.
    with torch.no_grad():
        logits = model(clip_tensor)
        # logits shape: (1, 12) -- one batch item, 12 word scores

    # Convert raw logits to probabilities using softmax.
    # softmax turns any list of numbers into probabilities that sum to 1.
    # dim=1 means "compute softmax across the 12 class scores" per item.
    probabilities = torch.softmax(logits, dim=1)

    # squeeze(0) removes the batch dimension: (1, 12) -> (12,)
    probabilities = probabilities.squeeze(0)

    # Get the index of the highest probability -- that's our prediction
    predicted_idx = torch.argmax(probabilities).item()
    # .item() converts a 1-element tensor to a plain Python number

    # Get the confidence score for the predicted word
    confidence = probabilities[predicted_idx].item()

    # Build the full scores dictionary for all 12 words
    # This lets the UI show a confidence bar for every word, not just
    # the top prediction -- makes for a much better demo
    all_scores = {
        IDX_TO_WORD[i]: round(probabilities[i].item(), 4)
        for i in range(NUM_CLASSES)
    }

    return {
        "predicted_word": IDX_TO_WORD[predicted_idx],
        "confidence": round(confidence, 4),
        "all_scores": all_scores
    }


# ---------------------------------------------------------------------------
# QUICK TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing inference with a dummy clip...\n")

    # Make a fake clip shaped like real data
    dummy_clip = np.random.randint(0, 255, (NUM_FRAMES, IMG_HEIGHT, IMG_WIDTH),
                                   dtype=np.uint8)

    result = predict_word(dummy_clip)

    print(f"Predicted word : {result['predicted_word']}")
    print(f"Confidence     : {result['confidence'] * 100:.1f}%")
    print(f"\nAll scores:")
    for word, score in sorted(result['all_scores'].items(),
                               key=lambda x: x[1], reverse=True):
        bar = "█" * int(score * 30)
        print(f"  {word:10s} {score:.4f}  {bar}")