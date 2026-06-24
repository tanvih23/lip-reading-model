from dataset import LipReadingDataset
from torch.utils.data import DataLoader

dataset = LipReadingDataset()

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True
)

for clips, labels in loader:

    print("Clips Shape:", clips.shape)
    print("Labels Shape:", labels.shape)

    break