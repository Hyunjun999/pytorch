import cv2, glob, os
from torch.utils.data import Dataset
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


class Fashionset(Dataset):
    def __init__(self, path, transform=None) -> None:
        super(Fashionset, self).__init__()
        self.label_dict = dict(
            zip(sorted(os.listdir(path)), range(len(os.listdir(path))))
        )
        self.data_dir = glob.glob(os.path.join(path, "*/*.png"))
        self.transform = transform

    def __getitem__(self, i):
        img_path = self.data_dir[i]
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Error loading image: {img_path}")
                return None, None
            img = np.array(img)
            k = img_path.split("/")[-2]
            label = self.label_dict[k]

            if self.transform:
                img = self.transform(image=img)["image"]

            return img, label
        except Exception as e:
            print(f"Error processing image: {img_path}, Error: {e}")
            return None, None

    def __len__(self):
        return len(self.data_dir)


# if __name__ == "__main__":
#     train_transform = A.Compose(
#         [
#             A.Normalize(),
#             A.Resize(640, 640),
#             A.HorizontalFlip(),
#             A.VerticalFlip(),
#             A.RandomRotate90(p=0.15),
#             A.Rotate(limit=90, p=0.3),
#             A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
#             ToTensorV2(),
#         ]
#     )
#     f = Fashionset("./cloth_category/train")
#     print(f.label_dict)
#     for i in f:
#         print(i)
