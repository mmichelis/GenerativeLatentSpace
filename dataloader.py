# ------------------------------------------------------------------------------
# Loads certain digits from MNIST for PyTorch network, other dataloaders as well
# ------------------------------------------------------------------------------

import torch
import torchvision


class MNIST (torch.utils.data.Dataset):
    """
    Loads n amount of training data of specified classes from one of the MNIST datasets (Digits, Fashion, Kuzushiji, EMNIST). MNIST data has shape [-1, 1, 28, 28].

    Arguments:
        dataset (torchvision.datasets) : one of the MNIST datasets transformed with ToTensor.
        labels (List): an integer list containing all the classes you want to have in the data.
        number_of_samples (int): how many of each class should appear.
        train (boolean): should it be training or testing dataset
    """
    def __init__(self, dataset, labels, number_of_samples=1000, train=True):
        super().__init__()
        self.data = []
        self.targets = []

        for label in labels:
            idx = (dataset.targets == label)
            if (idx.shape[0] == 0):
                print(f"ERROR: Label {label} not found!")
                continue

            # Originally data are bytes, we want floats between -1 and 1
            self.data.append(2*(dataset.data[idx][:number_of_samples].float() / 255)-1)
            self.targets.append(dataset.targets[idx][:number_of_samples])

        self.data = torch.cat(self.data, dim=0).view(-1, 1, 28, 28)
        # Typecast targets as longs.
        self.targets = torch.cat(self.targets, dim=0)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (self.data[idx], self.targets[idx])


class MNISTDigits (MNIST):
    """
    Loads n amount of training data of specified MNIST digits.

    Arguments:
        digits (List): an integer list containing all the digits you want to have in the data.
        number_of_samples (int): how many of each digit should appear.
        train (boolean): should it be training or testing dataset
    """
    def __init__(self, digits, number_of_samples=1000, train=True):
        dataset = torchvision.datasets.MNIST("Data/", train=train, download=True)
        
        super().__init__(dataset, digits, number_of_samples, train)


class FashionMNIST (MNIST):
    """
    Mappings: 
    Label	Description
    0	    T-shirt/top
    1	    Trouser
    2	    Pullover
    3	    Dress
    4	    Coat
    5	    Sandal
    6	    Shirt
    7	    Sneaker
    8	    Bag
    9	    Ankle boot
    """
    def __init__(self, labels, number_of_samples=1000, train=True):
        dataset = torchvision.datasets.FashionMNIST("Data/", train=train, download=True)
        
        super().__init__(dataset, labels, number_of_samples, train)


class EMNIST (MNIST):
    """
    Mappings (we use the "byclass" option): first 10 labels are the digits, then following are uppercase characters, and lastly lowercase characters.
    """
    def __init__(self, labels, number_of_samples=1000, train=True):
        # It seems that EMNIST data is tranposed.
        dataset = torchvision.datasets.EMNIST("Data/", 'byclass', train=train, download=True)
        dataset.data = dataset.data.transpose(-1,-2)
        
        super().__init__(dataset, labels, number_of_samples, train)


class CustomData (torch.utils.data.Dataset):
    """
    Transforms tensors of data X with labels y into a dataset. Allows labels to be None for test dataset.

    Arguments:
        X (torch.Tensor [N,D]) : inputs
        y (torch.Tensor [N]) : labels
    """
    def __init__(self, X, y):
        self.data = X
        self.targets = y

    def __len__(self):
        return len(self.data)
        

    def __getitem__(self, idx):
        return (self.data[idx], [] if self.targets is None else self.targets[idx])






