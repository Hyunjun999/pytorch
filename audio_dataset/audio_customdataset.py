from torch.utils.data import Dataset
from PIL import Image
import glob, os


class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        super(CustomDataset, self).__init__()
        self.data_dir = glob.glob(os.path.join(data_dir, "*", "*.png"))
        self.transform = transform
        self.label_dict = {"MelSpectrogram": 0, "STFT": 1, "waveshow": 2}

    def __getitem__(self, index):
        img_path = self.data_dir[index]
        img = Image.open(img_path)
        img = img.convert("RGB")
        label_name = img_path.split("/")[3]
        label = self.label_dict[label_name]

        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data_dir)


# c = CustomDataset("./audio_dataset/train/")
# print(len(c))
# for i in c:
#     print(i)
#     break
