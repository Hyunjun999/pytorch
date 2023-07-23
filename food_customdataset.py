import os
import cv2
import glob
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None) -> None:
        super().__init__()
        label = glob.glob(os.path.join(data_dir, "*"))
        label_dict = dict(
            # zip(sorted(map(lambda x: x.split("\\")[-1], label)), range(len(label))) # In Windows, have to use \\, but in unix and linux, we use /
            zip(sorted(map(lambda x: x.split("/")[-1], label)), range(len(label)))
        )
        self.data_dir = glob.glob(os.path.join(data_dir, "*", "*.jpg"))
        self.transform = transform
        self.label_dict = label_dict

    def __getitem__(self, index):
        img_path = self.data_dir[index]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        key = os.path.basename(os.path.dirname(img_path))
        label = self.label_dict[key]

        if self.transform:
            img = self.transform(image=img)["image"]

        return img, label

    def __len__(self):
        return self.data_dir


# c = CustomDataset('./food_dataset/train/')
# for i in c:
#     print(i)
#     break
