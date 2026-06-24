"""
train.py

THIS FILE'S JOB:
Take the untrained model from model.py (random weights, useless predictions)
and actually TEACH it by showing it thousands of examples, checking how wrong
it is each time, and nudging its weights in the direction that makes it
less wrong.

"""
from dataset import LipReadingDataset
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import LipReaderModel
from dummy_data import (
    get_dummy_batch,
    NUM_CLASSES,
    NUM_FRAMES,
    NUM_CHANNELS,
    IMG_HEIGHT,
    IMG_WIDTH,
)

USE_DUMMY_DATA = True
# True  -> train on fake random data (for testing the pipeline works)
# False -> train on real extracted GRID clips

DATA_DIR = "../data/processed"
# Where real data lives, once it exists. Structure expected:
#   data/processed/
#     bin/        <- folder named after each word
#       clip_001.npy
#       clip_002.npy
#       ...
#     blue/
#       clip_001.npy
#     ...
# Each .npy file = one video clip saved as a numpy array,
# shape (NUM_FRAMES, NUM_CHANNELS, IMG_HEIGHT, IMG_WIDTH)

BATCH_SIZE = 4
# How many video clips to process at once per training step.
# Larger = faster training but needs more GPU memory.
# 4 is safe for free Colab T4. Bump to 8 or 16 if you're not
# running out of memory.

NUM_EPOCHS = 20
# How many times to loop through the entire dataset.
# 20 is a reasonable starting point for dummy data testing.

LEARNING_RATE = 1e-3
# How big each weight "nudge" is per step. This is one of the most
# important numbers in all of ML.
# Too high -> model overshoots, loss bounces around and never settles
# Too low  -> model learns, but painfully slowly
# 0.001 (= 1e-3) is the standard "safe starting point" for Adam optimizer.

CHECKPOINT_DIR = "checkpoints"
# Folder where we save the best model weights during training.
# Person D's FastAPI backend will later load the file saved here.

# STEP 1: DEVICE SETUP
# PyTorch can run computations on two places:
#   CPU  -> your normal processor. Works everywhere. Slow for large models.
#   GPU  -> graphics card. 10-100x faster for matrix math (which is ALL
#            neural networks are under the hood). Required for real training.
#
# torch.cuda.is_available() checks if a GPU is accessible right now.
# On your local VSCode machine this will likely be False (no GPU = CPU).
# On Colab with GPU runtime enabled, this will be True.
#
# We store the result in `device` and then move ALL tensors and the model
# to that same device. PyTorch requires that data and model live on the
# SAME device -- mixing CPU tensors with a GPU model causes an immediate
# crash.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# You'll see "Using device: cpu" locally.
# You'll see "Using device: cuda" on Colab with GPU enabled. Good sign.

# STEP 2: DATA LOADING

def load_dummy_data():
    """
    Builds a fake dataset entirely in memory from random tensors.
    Used when USE_DUMMY_DATA = True.

    Returns a PyTorch DataLoader -- the standard object that handles
    batching, shuffling, and feeding data to the training loop.
    """
    # We'll make 64 fake video clips total -- enough to have multiple
    # batches per epoch so training feels like a real loop, not a
    # single step.
    NUM_DUMMY_CLIPS = 64
    videos = torch.randn(NUM_DUMMY_CLIPS, NUM_FRAMES, NUM_CHANNELS, IMG_HEIGHT, IMG_WIDTH)
    labels = torch.randint(0, NUM_CLASSES, (NUM_DUMMY_CLIPS,))

    # TensorDataset wraps tensors into a dataset object that DataLoader
    # understands. It pairs each video[i] with its label[i] automatically,
    # so when DataLoader grabs a batch it gets (video_batch, label_batch)
    # correctly paired together.
    dataset = TensorDataset(videos, labels)

    # DataLoader is PyTorch's standard "data feeder." It handles:
    #   batch_size -> how many clips to grab per step (defined above)
    #   shuffle    -> randomize clip order each epoch. Important: without
    #                 shuffling, the model sees the same sequence every
    #                 epoch and can accidentally learn the ORDER of examples
    #                 instead of the actual content. Always shuffle training
    #                 data.
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return loader


def load_real_data():
    dataset = LipReadingDataset(
        data_dir="data/train",
        labels_file="labels.json"
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    # Print vocab so you can confirm words loaded correctly
    print(f"  Vocab: {dataset.word_to_idx}")
    print(f"  Total clips: {len(dataset)}")
    return loader
# STEP 3: THE TRAINING LOOP

def train():
    """
    The main training function. Runs the full training loop and saves
    the best model checkpoint to disk.
    """

    # --- 3a. Load data ---
    print("Loading data...")
    if USE_DUMMY_DATA:
        train_loader = load_dummy_data()
        print(f"Loaded DUMMY data: {len(train_loader.dataset)} fake clips")
    else:
        train_loader = load_real_data()
        print(f"Loaded REAL data: {len(train_loader.dataset)} clips")

    # --- 3b. Build model and move it to device (CPU or GPU) ---
    model = LipReaderModel(num_classes=NUM_CLASSES)

    # .to(device) moves ALL of the model's internal tensors (weights,
    # biases) to whichever device we're using. After this line, the
    # model lives on GPU if available, CPU otherwise.
    model = model.to(device)
    print(f"Model created. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    # p.numel() = number of elements in a parameter tensor. 

    # --- 3c. Loss function ---
    # CrossEntropyLoss is THE standard loss for classification problems.
    # It takes:
    #   - logits: raw scores from the model (shape: batch_size, num_classes)
    #   - labels: the correct class index for each item (shape: batch_size)
    # And returns a single number: how wrong the model was, on average,
    # across the batch. 0 = perfect. Higher = worse.
    #
    # Internally it combines two steps:
    #   1. Softmax: converts raw scores into probabilities that sum to 1
    #   2. Negative log likelihood: penalizes the model heavily when it
    #      gave low probability to the CORRECT answer
    # You don't need to implement those manually -- CrossEntropyLoss
    # does both in one numerically stable operation.
    criterion = nn.CrossEntropyLoss()

    # --- 3d. Optimizer ---
    # The optimizer is the algorithm that actually UPDATES the weights
    # after each backward pass.
    #
    # Adam (Adaptive Moment Estimation) is the de facto standard optimizer
    # for deep learning in 2024. It's better than plain gradient descent
    # because it:
    #   - Keeps a "running average" of recent gradients (momentum) so it
    #     doesn't wildly change direction on noisy batches
    #   - Automatically adjusts the learning rate PER WEIGHT based on
    #     how much that weight has been updated recently -- weights that
    #     don't change much get bigger nudges, weights that change a lot
    #     get smaller nudges
    #
    # model.parameters() passes all trainable weights to the optimizer
    # so it knows what to update.
    # lr = learning rate (how big each nudge is) -- set above in config.
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 3e. Checkpoint setup ---
    # Create the checkpoints folder if it doesn't exist yet.
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    best_loss = float("inf")
    # float("inf") = infinity. We track the best (lowest) loss seen so far
    # and save the model whenever we beat it. This way the checkpoint file
    # always contains the BEST version of the model, not just the last one.
    
    # THE ACTUAL LOOP
    print("\nStarting training...\n")

    for epoch in range(NUM_EPOCHS):
        # epoch counts from 0 to NUM_EPOCHS-1.

        # model.train() switches the model into "training mode."
        # This matters because some layer types (like Dropout and
        # BatchNorm, which we might add later) behave DIFFERENTLY
        # during training vs. inference. Always call this at the
        # start of a training epoch, and model.eval() when evaluating.
        model.train()

        total_loss = 0.0     # accumulate loss across all batches this epoch
        correct = 0          # count how many predictions were correct
        total = 0            # count total predictions made

        for batch_idx, (videos, labels) in enumerate(train_loader):
            # train_loader yields one (videos, labels) pair per iteration.
            # videos shape: (BATCH_SIZE, NUM_FRAMES, NUM_CHANNELS, H, W)
            # labels shape: (BATCH_SIZE,)

            # Move this batch's data to the same device as the model.
            # MUST match the device the model is on, or PyTorch crashes.
            videos = videos.to(device)
            labels = labels.to(device)

            # ---- FORWARD PASS ----
            # Feed the videos through the model, get raw scores (logits).
            # logits shape: (BATCH_SIZE, NUM_CLASSES)
            logits = model(videos)

            # Calculate how wrong the predictions were.
            # `loss` is a single scalar tensor -- one number.
            loss = criterion(logits, labels)

            # ---- BACKWARD PASS ----
            # optimizer.zero_grad() clears any gradients left over from
            # the PREVIOUS batch. This is required because PyTorch
            # ACCUMULATES gradients by default (adds them up across
            # calls to .backward()). If you forget this, gradients from
            # 10 previous batches pile up and your weight updates become
            # completely wrong. Always zero before backward.
            optimizer.zero_grad()

            # loss.backward() tells PyTorch: "compute the gradient of
            # this loss with respect to every single trainable weight
            # in the model." PyTorch does this automatically using the
            # chain rule of calculus (backpropagation). You don't write
            # any calculus yourself -- this one line handles it all.
            loss.backward()

            # optimizer.step() uses those gradients to actually UPDATE
            # the weights. This is the moment the model literally gets
            # a little bit smarter (or at least less wrong).
            optimizer.step()

            # ---- TRACKING ----
            total_loss += loss.item()
            # .item() converts a 1-element tensor to a plain Python float.
            # Needed for printing/arithmetic -- you can't do normal math
            # with tensors directly in Python.

            # torch.argmax(logits, dim=1) finds which class index has
            # the highest score, per item in the batch.
            # dim=1 means "find the max across the 12 class scores"
            # (dimension 1), NOT across the batch (dimension 0).
            predicted = torch.argmax(logits, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        # ---- END OF EPOCH STATS ----
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total

        print(f"Epoch [{epoch+1:02d}/{NUM_EPOCHS}] "
              f"Loss: {avg_loss:.4f}  "
              f"Accuracy: {accuracy:.1f}%")

        # Save checkpoint if this is the best loss we've seen so far.
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")

            # torch.save saves any Python object to disk.
            # We save a dictionary containing:
            #   model_state_dict  -> all the learned weights (this is what
            #                        Person D's backend will load later)
            #   epoch             -> which epoch this was saved at (useful
            #                        for debugging / resuming training)
            #   loss              -> what the loss was (useful for records)
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "loss": best_loss,
            }, checkpoint_path)
            print(f"  -> Saved best model (loss: {best_loss:.4f})")

    print("\nTraining complete!")
    print(f"Best loss achieved: {best_loss:.4f}")
    print(f"Model saved to: {CHECKPOINT_DIR}/best_model.pth")

if __name__ == "__main__":
    train()