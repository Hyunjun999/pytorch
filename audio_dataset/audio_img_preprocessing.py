import glob
import os
import shutil
from sklearn.model_selection import train_test_split

folder_path = glob.glob(
    os.path.join("/Users/jun/Desktop/GitHub/pytorch/final_data/", "*")
)
folder_names = list(map(lambda x: x.split("/")[-1], folder_path))
print(folder_names)

for f in folder_names:
    os.makedirs(f"/Users/jun/Desktop/GitHub/pytorch/audio_dataset/{f}", exist_ok=1)
    path = glob.glob(
        os.path.join(
            f"/Users/jun/Desktop/GitHub/pytorch/final_data/{f}",
            "*",
            "*.png",
        )
    )
    for files in path:
        shutil.copy(
            files,
            f"/Users/jun/Desktop/GitHub/pytorch/audio_dataset/{f}",
        )


melspectrogram = glob.glob(
    os.path.join(
        "/Users/jun/Desktop/GitHub/pytorch/audio_dataset/MelSpectrogram", "*.png"
    )
)
stft = glob.glob(
    os.path.join("/Users/jun/Desktop/GitHub/pytorch/audio_dataset/STFT", "*.png")
)
waveshow = glob.glob(
    os.path.join("/Users/jun/Desktop/GitHub/pytorch/audio_dataset/waveshow", "*.png")
)

print(len(melspectrogram), len(stft), len(waveshow))

mel_train, mel_val = train_test_split(melspectrogram, test_size=0.2)
stft_train, stft_val = train_test_split(stft, test_size=0.2)
waveshow_train, waveshow_val = train_test_split(waveshow, test_size=0.2)


for s in ["MelSpectrogram", "STFT", "waveshow"]:
    os.makedirs(
        f"/Users/jun/Desktop/GitHub/pytorch/audio_dataset/train/{s}", exist_ok=True
    )
    os.makedirs(
        f"/Users/jun/Desktop/GitHub/pytorch/audio_dataset/val/{s}", exist_ok=True
    )

for t, v in zip(
    [mel_train, stft_train, waveshow_train], [mel_val, stft_val, waveshow_val]
):
    if t == mel_train and v == mel_val:
        for path in t:
            shutil.copy(
                path,
                os.path.join(
                    f"/Users/jun/Desktop/GitHub/pytorch/audio_dataset/train/MelSpectrogram"
                ),
            )
        for path in v:
            shutil.copy(
                path,
                os.path.join(
                    f"/Users/jun/Desktop/GitHub/pytorch/audio_dataset/val/MelSpectrogram"
                ),
            )
    elif t == stft_train and v == stft_val:
        for path in t:
            shutil.copy(
                path,
                os.path.join(
                    f"/Users/jun/Desktop/GitHub/pytorch/audio_dataset/train/STFT"
                ),
            )
        for path in v:
            shutil.copy(
                path,
                os.path.join(
                    f"/Users/jun/Desktop/GitHub/pytorch/audio_dataset/val/STFT"
                ),
            )
    elif t == waveshow_train and v == waveshow_val:
        for path in t:
            shutil.copy(
                path,
                os.path.join(
                    f"/Users/jun/Desktop/GitHub/pytorch/audio_dataset/train/waveshow"
                ),
            )
        for path in v:
            shutil.copy(
                path,
                os.path.join(
                    f"/Users/jun/Desktop/GitHub/pytorch/audio_dataset/val/waveshow"
                ),
            )
