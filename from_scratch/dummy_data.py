import torch
NUM_FRAMES=75
IMG_HEIGHT=64
IMG_WIDTH=64
NUM_CHANNELS=1
NUM_CLASSES=12

def get_dummy_batch(batch_size=4):
    """
    Generates a FAKE batch of lip-video data, shaped exactly like real data

    Returns:
        videos: a tensor (multi-dimensional array) of random pixel values,
                pretending to be `batch_size` separate video clips.
        labels: a tensor of random word-index labels (0 to 11), pretending
                to be the "correct answer" for each fake video.
    """
    videos=torch.randn(batch_size, NUM_FRAMES, NUM_CHANNELS,IMG_HEIGHT,IMG_WIDTH)
    labels = torch.randint(0, NUM_CLASSES, (batch_size,))

    return videos, labels
    if __name__ == "__main__":
        videos, labels = get_dummy_batch(batch_size=4)
        print("videos shape:", videos.shape)
        print("labels shape:", labels.shape)
        print("labels:", labels)
    