
from pathlib import Path

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


SEED = 42
BATCH_SIZE = 128

DATA_DIR = Path(__file__).resolve().parents[1] / "notebooks" / "data"


def get_data_loaders():
    transform = transforms.ToTensor()

    # Load MNIST training data
    full_train = datasets.MNIST(
        root=str(DATA_DIR),
        train=True,
        download=True,
        transform=transform
    )

    # Load the official test data separately
    test_dataset = datasets.MNIST(
        root=str(DATA_DIR),
        train=False,
        download=True,
        transform=transform
    )

    # Split training data reproducibly
    generator = torch.Generator().manual_seed(SEED)

    train_dataset, val_dataset = random_split(
        full_train,
        [54000, 6000],
        generator=generator
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_data_loaders()

    print("Training images:", len(train_loader.dataset))
    print("Validation images:", len(val_loader.dataset))
    print("Test images:", len(test_loader.dataset))

    images, labels = next(iter(train_loader))

    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    print("Pixel range:", images.min().item(), images.max().item())
