import tensorstore as ts
import numpy as np
import os
import json # For pretty printing attributes

def read_zarr_with_tensorstore(zarr_root_path, array_sub_path):
    print(f"Attempting to read Zarr array from: {os.path.join(zarr_root_path, array_sub_path)}")
    print(f"  (Please verify if 'sz3l' in '{array_sub_path}' is correct or if it should be 'sz3')")

    normalized_zarr_root_path = os.path.normpath(zarr_root_path)

    dataset_spec = {
        'driver': 'zarr3',
        'kvstore': {
            'driver': 'file',
            'path': normalized_zarr_root_path
        },
        'path': array_sub_path,
        'cache_pool': {'total_bytes_limit': 100000000},
    }

    try:
        print("Opening dataset with TensorStore...")
        dataset = ts.open(dataset_spec).result()
        print("Dataset opened successfully.")

        print(f"  Shape: {dataset.shape}")
        print(f"  Data Type: {dataset.dtype}")
        print(f"  Rank: {dataset.rank}")

        # ... (data reading part remains the same)
        if dataset.rank == 3:
            slice_shape = tuple(min(s, 2) for s in dataset.shape)
            if all(s > 0 for s in slice_shape):
                print(f"\nReading slice [0:{slice_shape[0]}, 0:{slice_shape[1]}, 0:{slice_shape[2]}]...")
                data_slice = dataset[0:slice_shape[0], 0:slice_shape[1], 0:slice_shape[2]].read().result()
                print("Data slice (first few elements if large):\n", data_slice)
            else:
                print("Cannot read slice, one or more dimensions are too small for a 2x2x2 slice.")
        elif dataset.rank > 0 :
             print(f"\nReading first element...")
             idx = tuple([0] * dataset.rank)
             first_element = dataset[idx].read().result()
             print("First element:\n", first_element)


        # 5. Access attributes (Corrected method)
        print("\nAccessing attributes...")
        # Get the full specification of the opened dataset as a Python dictionary.
        # Remove the incorrect 'include_metadata=True' argument.
        # 'include_defaults=True' can be used if you want to see all default values in the spec.
        dataset_json_spec = dataset.spec().to_json() # Corrected call

        # The Spec object in your error message shows a 'metadata' key.
        # Zarr attributes are expected to be within this 'metadata' dictionary,
        # under their own 'attributes' key, as per Zarr v3 spec.
        
        attributes = None
        if 'metadata' in dataset_json_spec:
            attributes = dataset_json_spec['metadata'].get('attributes')

        if attributes is not None:
            print("Attributes found:")
            print(json.dumps(attributes, indent=2))
        else:
            print("Attributes not found under 'metadata.attributes' in the dataset's JSON specification.")
            print("Full dataset JSON specification for inspection:")
            print(json.dumps(dataset_json_spec, indent=2))
            print("\nInspect the above JSON. User-defined Zarr attributes should be under the 'metadata' key,")
            print("and within that, under an 'attributes' key, if they were written correctly and read by TensorStore.")
            print("For example: spec['metadata']['attributes']['your_key']")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure:")
        print(f"  1. The Zarr store exists at '{normalized_zarr_root_path}'.")
        print(f"  2. The array sub-path '{array_sub_path}' (including 'sz3l' if intended) is correct.")
        # ... (other ensure points) ...

if __name__ == '__main__':
    # --- User Configuration ---
    zarr_root = '/lcrc/project/ECP-EZ/jp/git/arcnn/data/hurricane_v3_decompressed_sync.zarr'
    array_path_in_zarr = 'decompressed/1/CLOUD/sz3l/1e-03' # As from your error
    # --- End User Configuration ---

    if not os.path.exists(zarr_root):
         print(f"Error: Zarr root '{zarr_root}' not found. Please update the path.")
    else:
        read_zarr_with_tensorstore(zarr_root, array_path_in_zarr)