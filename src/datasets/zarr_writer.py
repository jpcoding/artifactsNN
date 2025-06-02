import zarr
from pathlib import Path

class ZarrDatasetWriter:
    def __init__(self, root_path, chunk_shape=(64, 64, 64),
                 compressor="zstd", clevel=3):
        self.root_path = Path(root_path)
        self.chunk_shape = chunk_shape
        self.zstore = zarr.open(self.root_path, mode='a')

    def _get_or_create_array(self, path, shape, dtype):
        if path in self.zstore:
            return self.zstore[path]
        return self.zstore.require_dataset(
            path, shape=shape, dtype=dtype,
            chunks=self.chunk_shape,
        )

    def add_original(self, field, timestep, data):
        path = f"original/{field}/{timestep}"
        arr = self._get_or_create_array(path, shape=data.shape, dtype=data.dtype)
        arr[:] = data

    def add_decompressed(self, field, timestep, compressor, eb, data, **metadata):
        path = f"decompressed/{compressor}/eb{eb}/{field}/{timestep}"
        arr = self._get_or_create_array(path, shape=data.shape, dtype=data.dtype)
        arr[:] = data
        arr.attrs.update({
            "compressor": compressor,
            "eb": eb,
            **metadata
        })
