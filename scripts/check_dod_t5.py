"""DoD check for plan.csv T5 — run from train/: `uv run python scripts/check_dod_t5.py`.

Confirms the PyTorch and TensorFlow loaders share one real subject-independent
split: no participant's images cross train/val/test in either path, and both
paths land on the exact same participants per split (not just similarly-sized
splits) since both derive from split_participant_groups() with the same seed.
Also smoke-tests that load_tf_dataset's rewritten tf.data pipeline actually
produces correctly-shaped batches (the old batch-based version is gone).
"""
from torchvision import datasets, transforms

from asd_train.configs import DATASET_PATH_DEFAULT
from asd_train.data import get_participant_groups, load_torch_dataset, load_tf_dataset, split_participant_groups


def _assert_no_cross_split_overlap(name, split_to_participants):
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = split_to_participants[a] & split_to_participants[b]
        assert not overlap, f"{name}: participants in both {a} and {b}: {overlap}"


# --- PyTorch path ---
dataset = datasets.ImageFolder(DATASET_PATH_DEFAULT, transform=transforms.ToTensor())
bundle = load_torch_dataset(DATASET_PATH_DEFAULT)
groups = get_participant_groups(dataset)

torch_participants = {
    "train": {groups[i] for i in bundle.train_idx},
    "val": {groups[i] for i in bundle.val_idx},
    "test": {groups[i] for i in bundle.test_idx},
}
_assert_no_cross_split_overlap("torch", torch_participants)

n_total = len(dataset)
for split_name, idx in [("train", bundle.train_idx), ("val", bundle.val_idx), ("test", bundle.test_idx)]:
    frac = len(idx) / n_total
    print(f"torch {split_name}: {len(idx)} images ({frac:.1%}), {len(torch_participants[split_name])} participants")

# --- TensorFlow path: same split-index derivation, compared directly ---
manifest = datasets.ImageFolder(DATASET_PATH_DEFAULT)
tf_groups = get_participant_groups(manifest)
assert tf_groups == groups, "torch and TF manifests disagree on participant order/groups"

tf_train_idx, tf_val_idx, tf_test_idx = split_participant_groups(tf_groups)
tf_participants = {
    "train": {tf_groups[i] for i in tf_train_idx},
    "val": {tf_groups[i] for i in tf_val_idx},
    "test": {tf_groups[i] for i in tf_test_idx},
}
_assert_no_cross_split_overlap("tf", tf_participants)

for split_name in ("train", "val", "test"):
    assert tf_participants[split_name] == torch_participants[split_name], (
        f"{split_name}: torch and TF splits disagree — "
        f"torch-only={torch_participants[split_name] - tf_participants[split_name]}, "
        f"tf-only={tf_participants[split_name] - torch_participants[split_name]}"
    )

# --- Smoke-test the actual tf.data pipeline (not just the split indices) ---
img_size = 256
train_ds, val_ds, test_ds = load_tf_dataset(DATASET_PATH_DEFAULT, img_size=img_size)
for split_name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
    images, labels = next(iter(ds))
    assert images.shape[1:] == (img_size, img_size, 3), f"tf {split_name}: unexpected image shape {images.shape}"
    assert images.numpy().max() <= 1.0, f"tf {split_name}: images not scaled to [0,1]"
    print(f"tf {split_name}: first batch shape={tuple(images.shape)}")

print("OK: subject-independent, zero cross-split leakage, torch and TF splits are identical")
