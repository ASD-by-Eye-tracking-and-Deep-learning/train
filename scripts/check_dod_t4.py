"""DoD check for plan.csv T4 — run from train/: `uv run python scripts/check_dod_t4.py`.

Confirms get_participant_groups parses every image's participant ID and that
the result lines up with the dataset (547 images, 54 participants with
images, no participant's images split across both class folders).
"""
from torchvision import datasets, transforms

from asd_train.configs import DATASET_PATH_DEFAULT
from asd_train.data import get_participant_groups

dataset = datasets.ImageFolder(DATASET_PATH_DEFAULT, transform=transforms.ToTensor())
groups = get_participant_groups(dataset)

assert len(groups) == len(dataset) == 547, f"expected 547 samples/groups, got {len(dataset)}/{len(groups)}"
assert len(set(groups)) == 54, f"expected 54 unique participants, got {len(set(groups))}"

targets = [label for _, label in dataset.samples]
by_participant: dict[int, set[int]] = {}
for group, target in zip(groups, targets):
    by_participant.setdefault(group, set()).add(target)
cross_class = [pid for pid, classes in by_participant.items() if len(classes) > 1]
assert not cross_class, f"participants with images in both classes: {cross_class}"

print("OK: 547 samples, 54 participants, no cross-class leakage")
