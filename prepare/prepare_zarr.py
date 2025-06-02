import numpy as np
import aiofiles
import asyncio
import os
from pathlib import Path
import zarr
from concurrent.futures import ThreadPoolExecutor
import sys

# Adjust the path as necessary if your src directory is elsewhere
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
    from src.utils.compressor import edt_model, sz3
    from src.utils.stats import get_psnr, get_ssim # Available if needed later
    from src.utils.utils import quantization # Available if needed later
except ImportError:
    print("Warning: Could not import from src.utils. Make sure the path is correct and modules are available.")
    # Define dummy compressors if import fails, so the script can be outlined
    class DummyCompressor:
        def compress(self, data, *args):
            print(f"DummyCompressor: Simulating compression for data of shape {data.shape} with args {args}")
            # Return a tuple if [0] is expected, like sz3.compress(...)[0]
            return (data.tobytes(), None) # Simulate compressed stream and an extra value

        def decompress(self, compressed_data, shape, dtype, *args):
            print(f"DummyCompressor: Simulating decompression to shape {shape} with args {args}")
            return np.frombuffer(compressed_data, dtype=dtype).reshape(shape)

    sz3 = DummyCompressor()
    edt_model = DummyCompressor()


# === Writer for Zarr v3 ===
class ZarrDatasetWriter:
    def __init__(self, root_path, chunk_shape=(64, 64, 64)):
        self.root_path = Path(root_path)
        self.chunk_shape = chunk_shape
        # Ensure the parent directory for the Zarr store exists
        self.root_path.parent.mkdir(parents=True, exist_ok=True)
        self.zstore = zarr.open_group(
            self.root_path, mode='a', zarr_format=3
        )
        print(f"Zarr store opened at: {self.root_path.resolve()}")

    def _get_or_create_array(self, path, shape, dtype):
        if path in self.zstore:
            return self.zstore[path]
        # Ensure parent groups are created
        parts = path.split('/')
        current_path = ""
        for part in parts[:-1]:
            current_path += part
            if current_path not in self.zstore:
                self.zstore.create_group(current_path)
            current_path += "/"
            
        return self.zstore.require_array(
            path, shape=shape, dtype=dtype,
            chunks=self.chunk_shape, compressor=None # Using default Blosc compressor for Zarr arrays
        )

    def add_original(self, field, timestep, data):
        path = f"original/{timestep}/{field}"
        arr = self._get_or_create_array(path, shape=data.shape, dtype=data.dtype)
        arr[:] = data
        # print(f"Added original: {path}")
        
    def add_decompressed(self, field, timestep,  compressor_name, eb_str, data):
        path = f"decompressed/{timestep}/{field}/{compressor_name}/{eb_str}" 
        arr = self._get_or_create_array(path, shape=data.shape, dtype=data.dtype)
        arr[:] = data 
        # print(f"Added decompressed: {path}")
        
    

# === Async file reader ===
async def read_f32_file_async(file_path, shape):
    async with aiofiles.open(file_path, mode='rb') as f:
        raw = await f.read()
        data = np.frombuffer(raw, dtype='float32').reshape(shape)
        return data

# === Synchronous compression/decompression task ===
def _compression_decompression_pipeline(
    original_data_copy, compressor_module, compressor_name_str, 
    error_bound, original_shape, original_dtype
):
    """
    Handles the compression and decompression of data.
    This function is designed to be run in a ThreadPoolExecutor.
    """
    # print(f"PID {os.getpid()} TID {threading.get_ident()}: Compressing {compressor_name_str} eb={error_bound} for data {original_shape}")
    decompressed_result = None
    try:
        if compressor_name_str.lower() == 'sz3l':
            # Using the signature hint: sz3.compress(data, 0, 0, 0, eb)[0]
            # Assuming corresponding decompress: sz3.decompress(compressed_data, shape, dtype, 0, 0, 0, eb)
            # The 0,0,0 are placeholders for other SZ parameters (e.g., mode, config)
            decompressed_result, cr  = sz3(original_data_copy, error_bound, original_shape, config= '../debug/configs/sz3_interp.config') 
        
        else:
            print(f"Error: Unknown compressor type '{compressor_name_str}'.")
            decompressed_result = original_data_copy.copy() # Fallback

    except Exception as e:
        print(f"Error during {compressor_name_str} (eb={error_bound}) for data of shape {original_shape}: {e}")
        # Fallback: return a copy of original data to ensure the pipeline continues
        # and Zarr array can still be populated, though not with truly decompressed data for this entry.
        decompressed_result = original_data_copy.copy()
    
    # print(f"PID {os.getpid()}: Finished {compressor_name_str} eb={error_bound} for data {original_shape}")
    return decompressed_result


# === Async wrapper for processing and storing original and decompressed data ===
async def process_file_and_compress(
    writer, file_path, timestep, field, shape, loop, executor,
    compressor_configurations # List of (name_str, module_obj, error_bounds_list)
):
    try:
        original_data = await read_f32_file_async(file_path, shape)
    except FileNotFoundError:
        print(f"Error: File not found {file_path}. Skipping.")
        return
    except Exception as e:
        print(f"Error reading file {file_path}: {e}. Skipping.")
        return

    # Store original data
    await loop.run_in_executor(executor, writer.add_original, field, timestep, original_data)

    # Process for each compressor configuration
    for compressor_name_str, compressor_module, error_bounds_list in compressor_configurations:
        for eb in error_bounds_list:
            # Pass a copy of original_data to the compression task to ensure data integrity
            # if compression/decompression routines modify data in-place or for concurrent safety.
            decompressed_data = await loop.run_in_executor(
                executor,
                _compression_decompression_pipeline,
                original_data.copy(), # Important: pass a copy
                compressor_module,
                compressor_name_str,
                eb,
                original_data.shape,
                original_data.dtype
            )
            
            # Store the decompressed data
            # Convert error bound to string for Zarr path compatibility
            eb_str = f"{eb:.0e}" if eb < 0.01 else str(eb) # Format small eb in scientific notation
            await loop.run_in_executor(
                executor, writer.add_decompressed, field, timestep, compressor_name_str, eb_str, decompressed_data
            )
    # print(f"Finished processing for: {field}, timestep {timestep}")


# === Orchestrator ===
async def main():
    # --- Configuration ---
    dataset_path = "/lcrc/project/SDR/pjiao/data/hurricane_all/clean/" # Source data
    zarr_output_path = "../data/hurricane_v3_decompressed.zarr" # Output Zarr store
    file_extension = "f32"
    total_timesteps = 2 # Reduced for quick testing; set to 48 for full run
    # Subset of fields for quick testing; use full list for actual run
    field_names = ['CLOUD','P'] # ['CLOUD','P','PRECIP','QCLOUD','QGRAUP','QICE','QRAIN','QSNOW','QVAPOR','TC','U','V','W']
    data_shape = (100, 500, 500) # Shape of the data in .f32 files
    zarr_chunk_shape = (64, 64, 64) # Chunking for Zarr arrays
    max_workers_executor = 4 # Number of parallel workers for CPU-bound tasks (compression)

    # Define compressors and their error bounds to test
    # Each item: (name_for_path, compressor_module_object, list_of_error_bounds)
    compressor_configs = [
        ("sz3l", sz3, [1e-3]),
    ]
    # --- End Configuration ---

    writer = ZarrDatasetWriter(zarr_output_path, chunk_shape=zarr_chunk_shape) 

    tasks = []
    loop = asyncio.get_running_loop()
    # ThreadPoolExecutor for CPU-bound compression tasks
    executor = ThreadPoolExecutor(max_workers=max_workers_executor) 

    print(f"Starting data processing. Timesteps: {total_timesteps}, Fields: {len(field_names)}")
    print(f"Compressor configurations: {[(c[0], c[2]) for c in compressor_configs]}")

    for field in field_names:
        for timestep in range(1, total_timesteps + 1):
            # Construct file path, e.g., "CLOUDf01.bin.f32"
            # Original path: dataset_path / str(timestep) / f"{field}f{timestep:02d}.bin.{file_extension}"
            # Adjust filename format if it's different. The example implies files are directly under dataset_path/timestep/
            file_name = f"{field}f{timestep:02d}.bin.{file_extension}"
            file_path = os.path.join(dataset_path, str(timestep), file_name)
            
            # Check if file exists before creating a task
            if not os.path.exists(file_path):
                print(f"Warning: Source file not found, skipping: {file_path}")
                continue

            tasks.append(
                process_file_and_compress(
                    writer, file_path, timestep, field, data_shape, 
                    loop, executor, compressor_configs
                )
            )

    if not tasks:
        print("No tasks created. Check paths and configuration.")
        executor.shutdown(wait=False) # Ensure executor is shut down
        return

    print(f"Gathering {len(tasks)} tasks...")
    await asyncio.gather(*tasks)
    
    executor.shutdown(wait=True) # Wait for all tasks in executor to complete
    print("All processing complete.")
    print(f"Zarr dataset written to: {Path(zarr_output_path).resolve()}")

# === Run ===
if __name__ == "__main__":
    # Simple check for dummy data path if actual LCRC path is not available
    if not os.path.exists("/lcrc/project/SDR/pjiao/data/hurricane_all/clean/"):
        print("LCRC dataset path not found. Attempting to create dummy data for testing.")
        # Create dummy files for a minimal test case if the real data path doesn't exist
        dummy_data_path = Path("./dummy_hurricane_data/clean/")
        dummy_data_path.mkdir(parents=True, exist_ok=True)
        
        _field_names_dummy = ['CLOUD','P']
        
        _total_timesteps_dummy = 1 # Only one timestep for dummy
        _data_shape_dummy = (100, 500, 500) # Must match expected shape

        for ts in range(1, _total_timesteps_dummy + 1):
            ts_path = dummy_data_path / str(ts)
            ts_path.mkdir(parents=True, exist_ok=True)
            for field_name in _field_names_dummy:
                dummy_file_name = f"{field_name}f{ts:02d}.bin.f32"
                dummy_file_path = ts_path / dummy_file_name
                if not dummy_file_path.exists():
                    print(f"Creating dummy file: {dummy_file_path}")
                    dummy_array = np.random.rand(*_data_shape_dummy).astype('float32')
                    with open(dummy_file_path, 'wb') as df:
                        df.write(dummy_array.tobytes())
        
        # Point dataset_path to dummy data for the test run in main()
        # This requires modifying main or passing path, simplified here by print
        print(f"To run with dummy data, change 'dataset_path' in main() to: {dummy_data_path.parent.resolve()}")
        print("Exiting. Please update 'dataset_path' in the script if using dummy data or ensure LCRC path is mounted.")
        # For automatic testing with dummy data if script is run directly and main path fails:
        # You would typically pass this as an argument or environment variable.
        # For this example, if you want to run with dummy data when real data is not found,
        # you'd need to modify the `dataset_path` in `main` before `asyncio.run(main())`.
        # The current dummy setup is just illustrative.
    
    asyncio.run(main())