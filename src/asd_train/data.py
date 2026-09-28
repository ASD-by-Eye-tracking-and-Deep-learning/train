"""Dataset loading for both the PyTorch (Ghost/CBAM) and TensorFlow (InceptionV3)
training paths. Both read the same ML4Autism scanpath images
(``mahmoud-dataset/Images``, two class subfolders) and now share ONE
subject-independent split (``split_participant_groups``, T5) — previously
they used two different, non-subject-independent splits (PyTorch: stratified
by image; TensorFlow: batch-based take/skip), which is exactly the data
leakage Reviewer 2 flagged. See ``ASD/.agents/record.md`` (T4/T5/Decision #6).
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from sklearn.model_selection import GroupShuffleSplit
from torchvision import datasets, transforms

_FILENAME_PARTICIPANT_PATTERN = re.compile(r"^(?:TS|TC)\d+_(\d+)\.(?:png|jpg|jpeg)$", re.IGNORECASE)


def get_participant_groups(dataset: datasets.ImageFolder) -> list[int]:
    """Participant ID per sample, in the same order as ``dataset.samples`` —
    i.e. aligned with the indices ``split_participant_groups`` returns, and
    with what ``StratifiedGroupKFold`` (T6) needs as its ``groups`` array.

    Filenames follow ``Class_ParticipantID`` (e.g. ``TS001_11.png`` ->
    participant 11), per ``mahmoud-dataset/ReadMe.txt``. Verified against
    ``Metadata_Participants.csv`` for all 547 images: parses cleanly, class
    (ASD/TD) always agrees with the CSV once IDs are compared as ints (the
    CSV doesn't zero-pad, filenames do), and no participant's images appear
    under both class folders. Parsed from the filename rather than joined
    against the CSV, since the CSV isn't needed for grouping and 5 of its 59
    participants have no images at all.
    """
    groups = []
    for path, _ in dataset.samples:
        match = _FILENAME_PARTICIPANT_PATTERN.match(Path(path).name)
        if not match:
            raise ValueError(f"Filename doesn't match Class_ParticipantID pattern: {path}")
        groups.append(int(match.group(1)))
    return groups


def split_participant_groups(
    groups: Sequence[int],
    val_size: float = 0.1845,
    test_size: float = 0.1155,
    random_state: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Subject-independent train/val/test split shared by both
    ``load_torch_dataset`` and ``load_tf_dataset`` (T5) — no participant's
    images appear in more than one split. Two-stage ``GroupShuffleSplit``,
    mirroring the *shape* of the old two-stage ``train_test_split`` it
    replaces (70/30, then the 30% split 0.385/0.615) but grouped by
    participant instead of stratified by class — ``plan.csv`` T5 specifies
    ``GroupShuffleSplit`` (not e.g. ``StratifiedGroupKFold``, which T6 adds
    separately for cross-validation). Defaults reproduce the original
    ~70/18.45/11.55% ratios.

    Returns indices into ``groups`` (i.e. into ``dataset.samples``).
    """
    n = len(groups)
    temp_size = val_size + test_size

    gss1 = GroupShuffleSplit(n_splits=1, test_size=temp_size, random_state=random_state)
    train_idx, temp_idx = next(gss1.split(range(n), groups=groups))

    temp_groups = [groups[i] for i in temp_idx]
    relative_test_size = test_size / temp_size
    gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_test_size, random_state=random_state)
    temp_val_local, temp_test_local = next(gss2.split(range(len(temp_idx)), groups=temp_groups))

    val_idx = [temp_idx[i] for i in temp_val_local]
    test_idx = [temp_idx[i] for i in temp_test_local]

    return list(train_idx), val_idx, test_idx


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
    """Subject-independent split via ``split_participant_groups`` (T5) — no
    participant's images appear in more than one of train/val/test. Same
    ~70/18.45/11.55% ratios all 4 Ghost/CBAM notebooks originally used, now
    grouped by participant instead of stratified by class/image.

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
    groups = get_participant_groups(dataset)
    train_idx, val_idx, test_idx = split_participant_groups(groups, random_state=random_state)

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


def load_tf_dataset(
    dataset_path: str,
    img_size: int = 256,
    batch_size: int = 32,
    val_size: float = 0.1845,
    test_size: float = 0.1155,
    random_state: int = 42,
):
    """Subject-independent split via the same ``split_participant_groups``
    (T5) the PyTorch path uses — replaces the old batch-based take/skip
    (``image_dataset_from_directory`` + ``.take()``/``.skip()``), which
    couldn't do arbitrary per-file subsetting and wasn't subject-independent
    or class-stratified either.

    Enumerates files via ``datasets.ImageFolder`` (no transform — just a
    manifest) so this shares the *exact* file order ``get_participant_groups``
    already relies on. With the same ``random_state`` as the PyTorch path,
    this assigns the identical set of participants to train/val/test as
    ``load_torch_dataset`` does — genuinely one shared split, not just two
    similarly-sized ones.

    ``img_size=256`` preserves ``image_dataset_from_directory``'s previous
    *implicit* default (never explicitly passed before this rewrite).
    """
    import tensorflow as tf

    manifest = datasets.ImageFolder(dataset_path)
    groups = get_participant_groups(manifest)
    train_idx, val_idx, test_idx = split_participant_groups(
        groups, val_size=val_size, test_size=test_size, random_state=random_state
    )

    def _load_and_preprocess(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_png(image, channels=3)  # drop alpha (source PNGs are RGBA)
        image = tf.image.resize(image, (img_size, img_size))
        image = image / 255.0
        return image, label

    def _make(indices):
        paths = [manifest.samples[i][0] for i in indices]
        labels = [manifest.samples[i][1] for i in indices]
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        return ds.batch(batch_size)

    return _make(train_idx), _make(val_idx), _make(test_idx)
