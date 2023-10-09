import torch
import torch.nn as nn


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()

        # convolutional layer 1
        # It sees 3x224x225 image tensor
        # and produces 16 feature maps of a tensor 16x224x224)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        # 2x2 pooling with stride 2. It sees tensors 16x224x224
        # and halves their size, i.e., the output will be 16x112x112
        self.pool1 = nn.MaxPool2d(2, 2)

        # convolutional layer 2
        # sees the output of the prev layer, i.e.,
        # 16x112x112 tensor
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)  # -> 32x112x112
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)  # -> 32x56x56

        # convolutional layer 3
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)  # -> 64x56x56
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)  # -> 64x28x28

        # convolutional layer 4
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)  # -> 128x28x28
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, 2)  # -> 128x14x14
    
        # convolutional layer 5
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)  # -> 256x14x14
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(2, 2)  # -> 256x7x7

        # linear layer (256 * 7 * 7 -> 12544)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256*7*7, 3136)
        self.dp1 = nn.Dropout(p=dropout)
        self.rl1 = nn.ReLU()

        # linear layer (3136 -> 1000(num_classes))
        self.fc2 = nn.Linear(3136, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.pool1(self.conv1(x)))
        x = self.relu2(self.pool2(self.conv2(x)))
        x = self.relu3(self.pool3(self.conv3(x)))
        x = self.relu4(self.pool4(self.conv4(x)))
        x = self.relu5(self.pool5(self.conv5(x)))

        x = self.flatten(x)

        x = self.rl1(self.dp1(self.fc1(x)))

        x = self.fc2(x)
        return x


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
