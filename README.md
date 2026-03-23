# CVDS Admission Assignments
## NHL Stenden Master Computer Vision & Data Science — 2026/2027

Solutions to the four debugging exercises required for the admission procedure of the Master CV&DS programme at NHL Stenden.

---

## Environment Setup

### Create and activate the Anaconda environment
```bash
conda env create -f environment.yml
conda activate admission-assignments
```

### Run any exercise
```bash
python exercise1.py
python exercise2.py
python exercise3.py
python exercise4.py
```

---

## Exercise Overview

### Exercise : Set Ordering Bug (exercise1.py)
**Bug:** Python sets have no guaranteed order. Iterating over a set produces elements in unpredictable sequence, making index-based fruit lookup unreliable.

**Fix:** Convert the set to a sorted list before indexing. A sorted list always produces the same alphabetical order, making the index reliable and consistent across all runs.

---

### Exercise 2 : NumPy Coordinate Swap Bug (exercise2.py)
**Bug 1 (Obvious):** Wrong column index on the right side of the assignment, `coords[:, 1]` was used twice instead of `coords[:, 0]`.

**Bug 2 (Hidden):** NumPy arrays are views into memory, not independent copies. Simultaneous assignment overwrites values in place, when column 0 is assigned first, its original values are lost before column 1 can receive them.

**Fix:** Make explicit `.copy()` of all columns before any assignment, so original values are preserved independently.

---

### Exercise 3 : CSV Data Type Bug (exercise3.py)
**Bug:** `csv.reader` returns all values as strings. After `np.stack`, the array contains string values. Matplotlib plots strings as categorical text sorted alphabetically, not numerically, so the precision-recall curve appears in the wrong order.

**Fix:** Convert the NumPy array to float immediately after stacking: `np.stack(results).astype(float)`

---

### Exercise 4 : GAN Training Bugs (exercise4.py)
**Bug 1 (Structural):** Label tensors created using the `batch_size` parameter instead of the actual current batch size. The last batch of each epoch often has fewer samples than `batch_size`, causing a size mismatch that crashes training with a ValueError.

**Fix:** Replace `batch_size` with `real_samples.size(0)` when creating label tensors, so they always match the actual batch.

**Bug 2 (Cosmetic):** Display condition `if n == batch_size - 1` ties visualisation frequency to the batch_size parameter, changing display behaviour whenever batch_size changes.

**Fix:** Replace with `if n == len(train_loader) - 1` to always trigger at the end of each epoch regardless of batch_size.

---

## Dependencies
- Python 3.10
- NumPy
- Matplotlib
- PyTorch
- Torchvision
- Pillow
- IPython
```
