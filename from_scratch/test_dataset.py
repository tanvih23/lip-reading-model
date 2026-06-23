from dataset import LipReadingDataset

dataset = LipReadingDataset()

print("Total Samples:", len(dataset))

clip, label = dataset[0]

print("Clip Shape:", clip.shape)
print("Label:", label)