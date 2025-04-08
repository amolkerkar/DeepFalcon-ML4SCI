#inspecting the dataset

import h5py
file_path = ".\\data\\quark-gluon_data-set_n139306.hdf5"

with h5py.File(file_path, 'r') as f:
    print("Top-level keys:")
    for key in f.keys():
        print(" -", key)
        
    #explore that one group
    def print_all(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"Dataset: {name}, shape: {obj.shape}")

    print("\nFull structure:")
    f.visititems(print_all)
