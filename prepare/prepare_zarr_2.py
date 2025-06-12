import numpy as np
import os
from pathlib import Path
import zarr
from zarr.errors import ContainsGroupError
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from src.utils.compressor import edt_model, sz3, cusz 
from src.utils.stats import get_psnr, get_ssim


class ZarrDatasetWriter:
    def __init__(self, root_path, chunk_shape=(64, 64, 64)):
        self.root_path = Path(root_path)
        self.chunk_shape = chunk_shape
        self.root_path.parent.mkdir(parents=True, exist_ok=True)
        self.zstore = zarr.open_group(self.root_path, mode='a', zarr_format=3)
        print(f"Zarr store opened at: {self.root_path.resolve()}")

    def _get_or_create_array(self, path, shape, dtype):
        if path in self.zstore:
            return self.zstore[path]
        parts = path.split('/')
        current_group_path = ""
        for part in parts[:-1]:
            current_group_path = os.path.join(current_group_path, part) if current_group_path else part
            if current_group_path not in self.zstore:
                try:
                    self.zstore.create_group(current_group_path)
                except ContainsGroupError:
                    pass
        return self.zstore.require_array(path, shape=shape, dtype=dtype, chunks=self.chunk_shape, compressors=None)

    def add_original(self, field, timestep, data, attributes_to_store=None):
        path = f"original/{timestep}/{field}"
        arr = self._get_or_create_array(path, shape=data.shape, dtype=data.dtype)
        arr[:] = data
        if isinstance(attributes_to_store, dict):
            arr.attrs.clear()
            arr.attrs.put(attributes_to_store)

    def add_decompressed(self, field, timestep, compressor_name, eb_str, data, attributes_to_store=None):
        path = f"decompressed/{timestep}/{field}/{compressor_name}/{eb_str}"
        arr = self._get_or_create_array(path, shape=data.shape, dtype=data.dtype)
        arr[:] = data
        if isinstance(attributes_to_store, dict):
            arr.attrs.clear()
            arr.attrs.put(attributes_to_store)


def read_f32_file_sync(file_path, shape):
    with open(file_path, mode='rb') as f:
        raw = f.read()
        return np.frombuffer(raw, dtype='float32').reshape(shape)


def _compression_decompression_pipeline(
    original_data_for_comparison,
    data_to_compress,
    compressor_module,
    compressor_name_str,
    error_bound,
    original_shape,
    original_dtype
):
    decompressed_result = None
    psnr_val = None
    ssim_val = None
    compression_successful = False
    cr = None
    try:
        if compressor_name_str.lower() == 'sz3l':
            decompressed_result, cr = sz3(original_data_for_comparison, error_bound, original_shape, config='../debug/configs/sz3_lorenzo.config') 
        elif compressor_name_str.lower() == 'edt':
            pass
        elif compressor_name_str.lower() == 'cusz':
            decompressed_result, cr = cusz(original_data_for_comparison, error_bound, original_shape)
        elif compressor_name_str.lower() == 'sz3i':
            decompressed_result, cr = sz3(original_data_for_comparison, error_bound, original_shape, config='../debug/configs/sz3_interp.config') 
        else:
            raise ValueError(f"Unknown compressor type '{compressor_name_str}'.")
        compression_successful = True
    except Exception as e:
        print(f"Error during {compressor_name_str} compression/decompression (eb={error_bound}) for shape {original_shape}: {e}")
        decompressed_result = data_to_compress.copy()
        compression_successful = False

    if decompressed_result is not None:
        data_range = np.ptp(original_data_for_comparison)
        if data_range == 0:
            data_range = 1.0
        try:
            psnr_val = get_psnr(original_data_for_comparison, decompressed_result)
            ssim_val = get_ssim(original_data_for_comparison, decompressed_result)
        except Exception as e:
            print(f"Error calculating PSNR/SSIM for {compressor_name_str} (eb={error_bound}): {e}")
            psnr_val = -1.0
            ssim_val = -1.0

    return {
        'data': decompressed_result,
        'psnr': psnr_val,
        'ssim': ssim_val,
        'compression_successful': compression_successful,
        'compression_ratio': cr
    }


def process_file_and_compress_sync(
    writer, file_path, timestep, field, shape,
    compress_executor,
    compressor_configurations, process_original=True 
):
    try:
        original_data = read_f32_file_sync(file_path, shape)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}. Skipping.")
        return f"Skipped: {file_path}"

    print(f"Processing {file_path}")
    if process_original:  
        writer.add_original(field, timestep, original_data, {'data_type': 'original', 'source_file': str(file_path)})

    compression_futures = []
    details = []
    for compressor_name_str, compressor_module, error_bounds_list in compressor_configurations:
        for eb in error_bounds_list:
            future = compress_executor.submit(
                _compression_decompression_pipeline,
                original_data,
                original_data.copy(),
                compressor_module,
                compressor_name_str,
                eb,
                original_data.shape,
                original_data.dtype
            )
            compression_futures.append(future)
            details.append({
                'future': future,
                'field': field,
                'timestep': timestep,
                'compressor_name': compressor_name_str,
                'numeric_eb': eb
            })

    future_map = {d['future']: d for d in details}
    for future in as_completed(compression_futures):
        d = future_map[future]
        try:
            result = future.result()
            eb_str = f"{d['numeric_eb']:.0e}" if d['numeric_eb'] < 0.01 else str(d['numeric_eb'])
            meta = {
                'data_type': 'decompressed',
                'source_field': d['field'],
                'source_timestep': d['timestep'],
                'original_source_file': str(file_path),
                'compression_parameters': {
                    'compressor_name': d['compressor_name'],
                    'error_bound_numeric': d['numeric_eb']
                },
                'processing_info': {
                    'compression_successful': result['compression_successful']
                }
            }
            if result['psnr'] not in [None, -1.0]:
                meta['psnr'] = result['psnr']
            if result['ssim'] not in [None, -1.0]:
                meta['ssim'] = result['ssim']
            if result['compression_ratio'] is not None:
                meta['cr'] = result['compression_ratio']
            writer.add_decompressed(
                d['field'], d['timestep'], d['compressor_name'], eb_str,
                result['data'], attributes_to_store=meta
            )
        except Exception as e:
            print(f"Compression task failed for {file_path}, {d['compressor_name']}: {e}")
    return f"Processed: {file_path}"


def main_sync():
    dataset_path_str = "/lcrc/project/SDR/pjiao/data/hurricane_all/clean/"
    zarr_output_path_str = "../data/hurricane.zarr"
    file_extension = "f32"
    total_timesteps = 48
    field_names = ['CLOUD','P','PRECIP','QCLOUD','QGRAUP','QICE','QRAIN','QSNOW','QVAPOR','TC','U','V','W']
    # total_timesteps = 1
    # field_names = ['CLOUD']

    data_shape = (100, 500, 500)
    zarr_chunk_shape = (64, 64, 64)
    # compressor_configs = [("cusz", cusz, [5e-3])]
    compressor_configs = [
        ("sz3i", sz3, [1e-2]),
    ]
    process_original = False

    field_names = [f.upper() for f in field_names]
    dataset_path = Path(dataset_path_str)
    writer = ZarrDatasetWriter(zarr_output_path_str, chunk_shape=zarr_chunk_shape)

    file_executor = ThreadPoolExecutor(max_workers=64)
    compress_executor = ProcessPoolExecutor(max_workers=1)

    tasks = []
    for field in field_names:
        for timestep in range(1, total_timesteps + 1):
            file_name = f"{field}f{timestep:02d}.bin.{file_extension}"
            file_path = dataset_path / str(timestep) / file_name
            if not file_path.exists():
                print(f"Skipping missing file: {file_path}")
                continue
            tasks.append(
                file_executor.submit(
                    process_file_and_compress_sync,
                    writer, file_path, timestep, field, data_shape,
                    compress_executor, compressor_configs, process_original
                )
            )

    for i, future in enumerate(as_completed(tasks)):
        try:
            print(future.result())
        except Exception as e:
            print(f"Task {i + 1} failed: {e}")

    file_executor.shutdown(wait=True)
    compress_executor.shutdown(wait=True)
    print("All processing complete.")
    print(f"Zarr dataset written to: {Path(zarr_output_path_str).resolve()}")


if __name__ == "__main__":
    main_sync()
