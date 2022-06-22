from torch.utils.data import dataset


class FMnist(dataset.Dataset):
    def __init__(self, path="/home/guo/myDisk/Dataset/fashion-mnist", train=True):
        super(FMnist, self).__init__()
        if train is True:
            self.images, self.labels = load_mnist(path, 'train')
        else:
            self.images, self.labels = load_mnist(path, 't10k')
        self.len = self.images.shape[0]

    def __getitem__(self, key):
        if type(key) != int:
            print("getitem index: ", key)
            return

        index = key % self.len
        img = self.images[index].reshape((1, 28, 28))
        label = self.labels[index]
        return img, label

    def __len__(self):
        return self.len


# from fashion-mnist/utils/mnist_reader.py
def load_mnist(path="/home/guo/myDisk/Dataset/fashion-mnist", kind='train'):
    import os
    import gzip
    import numpy as np

    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    return images, labels


if __name__ == "__main__":
    imgs, labels = load_mnist()
    print(imgs.shape)
    print(labels.shape)

    fMnist = FMnist()
    # image & label
    print(fMnist[10])

    # test __len__ attributes
    print(len(fMnist))