"""Dataset loading for both the PyTorch (Ghost/CBAM) and TensorFlow (InceptionV3)
training paths. Both read the same ML4Autism scanpath images
(``mahmoud-dataset/Images``, two class subfolders) but split them differently,
matching what each original notebook actually did — not unified, since
changing the split would change results.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import torch
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms


@dataclass
class TorchDataBundle:
    dataset: datasets.ImageFolder
    train_loader: torch.utils.data.DataLoader
    val_loader: torch.utils.data.DataLoader
    test_loader: torch.utils.data.DataLoader
    train_idx: list
    val_idx: list
    test_idx: list


def load_torch_dataset(
    dataset_path: str,
    img_size: int = 224,
    batch_size: int = 32,
    extra_transform_augmentation: bool = False,
    random_state: int = 42,
) -> TorchDataBundle:
    """70/30 split, then the 30% split 20/10 (test_size=0.385), both
    stratified by class — matches all 4 Ghost/CBAM notebooks.

    ``extra_transform_augmentation``: only ``ghost/mobilenetv4_small`` added
    RandomHorizontalFlip/RandomRotation/ColorJitter directly in the
    torchvision transform, *on top of* the separate ``augment_pre_fmmix1``
    step applied later in the training loop — the other 3 notebooks didn't.
    This looks like an inconsistency in the original work, not a deliberate
    choice, but is preserved here rather than silently "fixed".
    """
    transform_steps = [transforms.Resize((img_size, img_size))]
    if extra_transform_augmentation:
        transform_steps += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    transform_steps.append(transforms.ToTensor())
    transform = transforms.Compose(transform_steps)

    dataset = datasets.ImageFolder(dataset_path, transform=transform)
    targets = [label for _, label in dataset]

    train_idx, temp_idx = train_test_split(
        range(len(dataset)), test_size=0.3, stratify=targets, random_state=random_state
    )
    temp_targets = [targets[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.385, stratify=temp_targets, random_state=random_state
    )

    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)

    return TorchDataBundle(
        dataset=dataset,
        train_loader=torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler),
        val_loader=torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=val_sampler),
        test_loader=torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_sampler),
        train_idx=list(train_idx),
        val_idx=list(val_idx),
        test_idx=list(test_idx),
    )


def class_counts(indices, dataset) -> Counter:
    labels = [dataset.samples[i][1] for i in indices]
    return Counter(labels)


def load_tf_dataset(dataset_path: str, train_batches: int = 12, val_batches: int = 3, test_batches: int = 2):
    """Batch-based split used by the InceptionV3 notebook: loads the whole
    directory as a batched dataset, scales to [0,1], then takes/skips whole
    *batches* (not stratified by class) for train/val/test."""
    import tensorflow as tf

    data = tf.keras.utils.image_dataset_from_directory(dataset_path)
    data = data.map(lambda x, y: (x / 255, y))

    train = data.take(train_batches)
    val = data.skip(train_batches).take(val_batches)
    test = data.skip(train_batches + val_batches).take(test_batches)
    return train, val, test
