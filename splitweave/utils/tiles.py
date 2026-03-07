from collections import defaultdict
import os

def get_tile_content(directory, mode="train", *args, **kwargs):
    files = os.listdir(directory)
    # filenames = [os.path.join(DIR, f) for f in files]
    filenames = [os.path.join(directory, f.split("_")[0]) for f in files]
    filename_to_indices = defaultdict(list)
    for i, f in enumerate(files):
        name = filenames[i]
        ind = int(f.split("_")[-1].split(".")[0])
        filename_to_indices[name].append(ind)
    lens = []
    for key, value in filename_to_indices.items():
        filename_to_indices[key] = sorted(value)
    if mode == "train":
        train_limit = int(len(filenames) * 0.9)
        filenames = filenames[:train_limit]
    elif mode == "val":
        train_limit = int(len(filenames) * 0.9)
        filenames = filenames[train_limit:]
    return filenames, filename_to_indices