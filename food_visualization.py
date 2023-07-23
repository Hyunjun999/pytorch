import os
import matplotlib.pyplot as plt


class DataVisualization:
    def __init__(self, data_dir) -> None:
        self.data_dir = data_dir
        self.all_data = dict()
        self.train_data = dict()
        self.val_data = dict()
        self.test_data = dict()

    def load_data(self):
        train = os.path.join(self.data_dir, "train")
        val = os.path.join(self.data_dir, "validation")
        test = os.path.join(self.data_dir, "test")

        for label in os.listdir(train):
            label_dir = os.path.join(train, label)
            print(label_dir)
            cnt = len(os.listdir(label_dir))
            self.all_data[label] = cnt
            self.train_data[label] = cnt

        for label in os.listdir(val):
            label_dir = os.path.join(val, label)
            print(label_dir)
            cnt = len(os.listdir(label_dir))
            self.val_data[label] = cnt
            if label in self.all_data:
                self.all_data[label] += cnt
            else:
                self.all_data[label] = cnt

        for label in os.listdir(test):
            label_dir = os.path.join(test, label)
            cnt = len(os.listdir(label_dir))
            print(label_dir)
            self.test_data[label] = cnt
            if label in self.all_data:
                self.all_data[label] += cnt
            else:
                self.all_data[label] = cnt

    def visualiza_data(self):
        label = list(self.all_data.keys())
        cnt = list(self.all_data.values())
        print(label, cnt)

        plt.figure(figsize=(10, 6))
        plt.bar(label, cnt)
        plt.title("Label and Data size")
        plt.xlabel("Label")
        plt.ylabel(" # of Data")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.show()


if __name__ == "__main__":
    v = DataVisualization("./food_dataset/")
    v.load_data()
    v.visualiza_data()
