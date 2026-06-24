import torch
import torch.nn as nn

from dummy_data import get_dummy_batch, NUM_FRAMES, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS, NUM_CLASSES


class LipReaderCNN(nn.Module):
    """
    The CNN half. Its ONLY job: take ONE frame (one 64x64 grayscale image)
    and compress it down into a short list of numbers (a "feature vector")
    that captures what matters about that frame's mouth shape.

    "nn.Module" is PyTorch's base class for anything that's a piece of a
    neural network. Inheriting from it gives us a lot of built-in machinery
    (tracking trainable weights, moving the model to GPU, saving/loading,
    etc.) for free -- you don't write that machinery yourself.
    """

    def __init__(self):
        super().__init__()
        # Conv2d(in_channels, out_channels, kernel_size, padding)
        #   in_channels  -> how many channels the INPUT has (1 = grayscale)
        #   out_channels -> how many DIFFERENT filters to learn at this layer
        #   kernel_size  -> the size of the sliding filter window (3x3 is a
        #                   very common, small, efficient choice)
        #   padding      -> adds a thin border of zeros around the image so
        #                   the output size doesn't shrink after convolving.
        #                   padding=1 with kernel_size=3 keeps height/width
        #                   the same as the input.
        self.conv1 = nn.Conv2d(in_channels=NUM_CHANNELS, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # ReLU = Rectified Linear Unit. It's an "activation function" --
        # after a convolution does its math (which is just weighted sums,
        # fundamentally linear/straight-line math), ReLU adds non-linearity
        # by doing: if a number is negative, make it 0; otherwise leave it
        # alone.
        self.relu = nn.ReLU()
        # MaxPool2d shrinks the image by taking the LARGEST value in each
        # small window (2x2 here) and discarding the rest. Two purposes:
        #   1. Reduces image size -> less computation in later layers.
        #   2. Keeps the STRONGEST signal in each region while discarding
        #      exact pixel position -- gives the network some tolerance to
        #      the mouth being a couple pixels off-center between frames.
        self.pool = nn.MaxPool2d(kernel_size=2)
        #We need to know the final flattened size after two conv+pool
        # rounds, to wire up the next layer correctly. Starting at 64x64:
        #   after pool 1: 64 / 2 = 32x32
        #   after pool 2: 32 / 2 = 16x16
        # Final feature maps: 32 channels (from conv2) * 16 * 16 spatial size
        self.flattened_size = 32 * 16 * 16

        # A "Linear" layer (a.k.a. fully connected / dense layer) takes a
        # flat list of numbers in and produces a flat list of numbers out,
        # where EVERY input number influences EVERY output number via a
        # learnable weight. Here we compress our big flattened feature map
        # down to a tidy 128-number summary -- this 128 is our "feature
        # vector" for one single frame, mentioned in the big-picture diagram
        # above.
        self.fc = nn.Linear(self.flattened_size, 128)

    def forward(self, x):
        """
        "forward" defines what happens when data actually flows through this
        module. PyTorch calls this automatically when you do `model(input)`.

        Input x shape: (batch_size, NUM_CHANNELS, IMG_HEIGHT, IMG_WIDTH)
        -- i.e. a batch of SINGLE frames (we'll feed it one frame at a time
        per video, looped over in the bigger model below).
        """
        x = self.conv1(x)      # extract first-level patterns (edges, curves)
        x = self.relu(x)       # add non-linearity
        x = self.pool(x)       # shrink: 64x64 -> 32x32

        x = self.conv2(x)      # extract second-level, more complex patterns
        x = self.relu(x)
        x = self.pool(x)       # shrink: 32x32 -> 16x16
         # .view() reshapes a tensor without changing its data -- here we
        # flatten the (channels, height, width) 3D feature map into one
        # long 1D list per item in the batch, so the Linear layer (which
        # expects flat input) can use it.
        # x.size(0) means "keep the batch size dimension as-is, whatever it
        # is" -- we let PyTorch infer it rather than hardcoding it.
        x = x.view(x.size(0), -1)

        x = self.fc(x)         # compress down to a 128-number feature vector
        return x
class LipReaderModel(nn.Module):
    """
    THE FULL MODEL. Wraps the per-frame CNN above, runs it across every
    frame in a video, feeds that sequence into an LSTM, and produces a
    final word prediction.
    """

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        # One CNN, reused for EVERY frame. We don't create 75 separate CNNs --
        # the same CNN (the same learned filters) looks at each frame, using the same visual
        # knowledge whether it's frame 1 or frame 50. Sharing weights across
        # time like this is standard and keeps the model far smaller.
        self.cnn = LipReaderCNN()
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)

        # Final classifier: takes the LSTM's summary (128 numbers) and
        # produces one raw score per possible word (12 scores). These raw
        # scores are called "logits" -- not yet probabilities, just relative
        # scores where higher = the model thinks this word is more likely.
        self.classifier = nn.Linear(128, num_classes)
    def forward(self, video):
        """
        Input video shape: (batch_size, NUM_FRAMES, NUM_CHANNELS, H, W)
        -- exactly what get_dummy_batch() produces, and what Person A's
        real pipeline will eventually produce too.

        Output shape: (batch_size, NUM_CLASSES) -- one score per word,
        per video in the batch.
        """
        batch_size, num_frames, channels, height, width = video.shape
        # STEP 1: run the CNN on every single frame.
        # We can't feed a whole (batch, frames, channels, H, W) 5D tensor
        # into a CNN -- Conv2d expects 4D input: (batch, channels, H, W).
        # So we temporarily MERGE the batch and frame dimensions together,
        # run the CNN once on all of them at once (much faster than a
        # Python for-loop over 75 frames), then split them back apart.
        x = video.view(batch_size * num_frames, channels, height, width)

        # Run the shared CNN on this giant stack of individual frames.
        # Output shape: (batch_size * num_frames, 128)
        frame_features = self.cnn(x)
        # Now split batch and frames back into two separate dimensions,
        # restoring the sequence structure the LSTM needs.
        # Shape becomes: (batch_size, num_frames, 128)
        frame_features = frame_features.view(batch_size, num_frames, -1)
        # STEP 2: feed the sequence of per-frame features into the LSTM.
        # The LSTM processes the sequence step by step internally (frame 1,
        # then frame 2, ..., updating its memory each time) and returns:
        #   lstm_out    -> the LSTM's output at EVERY single time step
        #                  
        #   (h_n, c_n)  -> the FINAL hidden state and cell state after
        #                  seeing the whole sequence.
        lstm_out, (h_n, c_n) = self.lstm(frame_features)
        # h_n shape is technically (num_layers, batch_size, hidden_size).
        # We only have 1 LSTM layer, so h_n[0] gives us shape
        # (batch_size, hidden_size) -- one 128-number summary per video.
        final_summary = h_n[0]

        # STEP 3: turn that summary into a prediction over our 12 words.
        logits = self.classifier(final_summary)

        return logits

if __name__ == "__main__":
    videos, labels = get_dummy_batch(batch_size=4)
    print("Input video batch shape:", videos.shape)

    model = LipReaderModel(num_classes=NUM_CLASSES)
    output = model(videos)

    print("Output shape:", output.shape)
    print("Expected shape: (4, 12)  <- 4 videos in batch, 12 word scores each")
    print()
    print("Raw output for first video in batch (12 scores, one per word):")
    print(output[0])
