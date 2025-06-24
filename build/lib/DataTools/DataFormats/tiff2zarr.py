import os
import shutil
import numpy as np
import zarr
import click
import json
from numcodecs import Blosc
from .utils import calculate_global_min_max, load_tiff_chunked, downsample, minmaxHisto
from .log import info, setup_custom_logger

def calculate_levels(data):

   get_divisions = lambda n: (n & -n).bit_length() - 1
   dim = []
   sh = data.shape
   for i in range(0,len(sh)):
	   dim.append(get_divisions(sh[i]))

   return min(dim)


def save_zarr(volume, output_path, chunks, compression, pixel_size, mode='w', original_dtype=np.uint8):
    store = zarr.DirectoryStore(output_path)
    compressor = Blosc(cname=compression, clevel=1, shuffle=2)

    if mode == 'w':
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        root_group = zarr.group(store=store)
    else:
        root_group = zarr.open(store=store, mode='a')

    levels = calculate_levels(volume)

    if levels > 6:
        levels = 6

    pyramid_levels, _ = downsample(volume, levels)
    datasets = []

    #shift index
    j = 0
    for level, data in enumerate(pyramid_levels):
        data = data.astype(original_dtype)

        dataset_name = f"{level}"
        if dataset_name in root_group:
            z = root_group[dataset_name]
            z.append(data, axis=0)
        else:
            z = root_group.create_dataset(
                name=dataset_name, shape=data.shape, chunks=chunks, dtype=data.dtype, compressor=compressor
            )
            z[:] = data

        scale_factor = 2 ** level

        datasets.append({
            "path": dataset_name,
            "coordinateTransformations": [
                {"type": "scale", "scale": [scale_factor, scale_factor, scale_factor]},
                {"type": "translation", "translation": [2**(level-1) - 0.5, 2**(level-1) - 0.5, 2**(level-1) - 0.5]}
            ]
        })
        j += 1

    root_group.attrs["multiscales"] = [{
        "version": "0.4",
        "name": "example",
        "axes": [
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"}
        ],
        "datasets": datasets,
        "type": "gaussian",
        "metadata": {
            "method": "skimage.transform.downscale_local_mean",
            "version": "0.16.1",
            "args": "[true]",
            "kwargs": {"anti_aliasing": True, "preserve_range": True}
        }
    }]


@click.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--dtype', type=click.Choice(['int8', 'int16', 'int32', 'uint8', 'uint16', 'float32', 'float64']), default='uint8', help='Data type of the images.')
@click.option('--chunks', type=(int, int, int), default=(64, 64, 64), help='Chunk size for the Zarr array as a tuple of three integers.')
@click.option('--compression', type=click.Choice(['blosclz', 'lz4', 'lz4hc', 'zlib', 'zstd']), default='blosclz', help='Compression algorithm to use for Zarr storage.')
@click.option('--pixel_size', type=float, default=1.0, help='Pixel size in micrometers.')
@click.option('--chunk_size', type=int, default=64, help='Number of TIFF images to load in each chunk.')
@click.option('--verbose', is_flag=True, help='Enable verbose logging.')
def main(input_dir, output_path, dtype, chunks, compression, pixel_size, chunk_size, verbose):
    """
    Main function to process TIFF images and save them as a Zarr store with multiscale representations.

    Parameters:
    - input_dir (str): Path to the input directory containing TIFF images.
    - output_path (str): Path to the output Zarr store.
    - dtype (str): Data type of the images. Choices are 'int8', 'int16', 'int32', 'uint8', 'uint16', 'float32', 'float64'.
    - chunks (tuple of ints): Chunk size for the Zarr array as a tuple of three integers.
    - compression (str): Compression algorithm to use for Zarr storage. Choices are 'blosclz', 'lz4', 'lz4hc', 'zlib', 'zstd'.
    - pixel_size (float): Pixel size in micrometers.
    - chunk_size (int): Number of TIFF images to load in each chunk.
    - verbose (bool): Enable verbose logging.

    Returns:
    - None
    """
    setup_custom_logger(verbose=verbose)
    
    dtype_map = {'int8': np.int8, 'int16': np.int16, 'int32': np.int32, 'uint8': np.uint8, 'uint16': np.uint16, 'float32': np.float32, 'float64': np.float64}
    global_min, global_max = minmaxHisto(input_dir)
    #global_min, global_max = calculate_global_min_max(input_dir)
    info(f"Global min and max found: {global_min}, dtype: {global_max}")
    start_index = 0
    mode = 'w'

    while True:
        stack, start_index = load_tiff_chunked(input_dir, dtype_map[dtype], chunk_size, start_index, global_min, global_max)
        if stack is None:
            break
        save_zarr(stack, output_path, chunks, compression, pixel_size, mode, original_dtype=dtype_map[dtype])
        mode = 'a'
        if start_index >= len(os.listdir(input_dir)):
            break

if __name__ == "__main__":
    main()
