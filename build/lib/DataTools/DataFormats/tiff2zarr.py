import os
import shutil
import numpy as np
import zarr
import click
from zarr.codecs import BloscCodec, ShardingCodec, BytesCodec
from zarr.storage import LocalStore
from .utils import load_tiff_chunked, downsample, minmaxHisto
from .log import info, setup_custom_logger

def calculate_levels(data):
    """Calculate the number of pyramid levels based on data shape."""
    get_divisions = lambda n: (n & -n).bit_length() - 1
    dim = []
    sh = data.shape
    for i in range(0, len(sh)):
        dim.append(get_divisions(sh[i]))
    return min(dim)


def save_zarr(volume, output_path, chunks, compression, pixel_size, shard_shape=None, mode='w', original_dtype=np.uint8):
    """
    Save a 3D volume to a Zarr v3 store with sharding, creating a multiscale pyramid representation.

    Parameters:
    - volume (numpy array): The 3D volume data to be saved.
    - output_path (str): The path to the output Zarr store.
    - chunks (tuple of ints): The chunk size for the Zarr array.
    - compression (str): The compression algorithm to use (e.g., 'blosclz', 'lz4', etc.).
    - pixel_size (float): The size of the pixels in micrometers.
    - shard_shape (tuple of ints, optional): The shard shape. If None, no sharding is used.
    - mode (str, optional): The mode to open the Zarr store ('w' for write, 'a' for append). Default is 'w'.
    - original_dtype (numpy dtype, optional): The original data type of the images. Default is np.uint8.

    Returns:
    - None
    """
    store = LocalStore(output_path)

    if mode == 'w':
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        root_group = zarr.group(store=store, overwrite=True)
    else:
        root_group = zarr.open_group(store=store, mode='a')

    levels = calculate_levels(volume)

    if levels > 6:
        levels = 6

    pyramid_levels = downsample(volume, levels)
    datasets = []

    for level, data in enumerate(pyramid_levels):
        data = data.astype(original_dtype)

        dataset_name = f"{level}"
        
        # Setup codecs chain
        codecs = [
            BytesCodec(),
            BloscCodec(cname=compression, clevel=5, shuffle='shuffle')
        ]
        
        # Add sharding if specified
        if shard_shape is not None:
            codecs = [
                ShardingCodec(
                    chunk_shape=chunks,
                    codecs=codecs
                )
            ]
        
        if dataset_name in root_group:
            z = root_group[dataset_name]
            # For append mode in v3, we need to read, combine, and recreate
            old_data = z[:]
            combined_data = np.concatenate([old_data, data], axis=0)
            
            # Remove old array and create new one with combined data
            del root_group[dataset_name]
            
            z = root_group.create_array(
                name=dataset_name,
                shape=combined_data.shape,
                chunks=shard_shape if shard_shape is not None else chunks,
                dtype=original_dtype,
                codecs=codecs
            )
            z[:] = combined_data
        else:
            z = root_group.create_array(
                name=dataset_name,
                shape=data.shape,
                chunks=shard_shape if shard_shape is not None else chunks,
                dtype=original_dtype,
                codecs=codecs
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
@click.option('--shard_shape', type=(int, int, int), default=None, help='Shard shape as a tuple of three integers. If not specified, no sharding is used.')
@click.option('--compression', type=click.Choice(['blosclz', 'lz4', 'lz4hc', 'zlib', 'zstd']), default='blosclz', help='Compression algorithm to use for Zarr storage.')
@click.option('--pixel_size', type=float, default=1.0, help='Pixel size in micrometers.')
@click.option('--chunk_size', type=int, default=64, help='Number of TIFF images to load in each chunk.')
@click.option('--verbose', is_flag=True, help='Enable verbose logging.')
def main(input_dir, output_path, dtype, chunks, shard_shape, compression, pixel_size, chunk_size, verbose):
    """
    Main function to process TIFF images and save them as a Zarr v3 store with multiscale representations and optional sharding.

    Parameters:
    - input_dir (str): Path to the input directory containing TIFF images.
    - output_path (str): Path to the output Zarr store.
    - dtype (str): Data type of the images. Choices are 'int8', 'int16', 'int32', 'uint8', 'uint16', 'float32', 'float64'.
    - chunks (tuple of ints): Chunk size for the Zarr array as a tuple of three integers.
    - shard_shape (tuple of ints, optional): Shard shape as a tuple of three integers.
    - compression (str): Compression algorithm to use for Zarr storage. Choices are 'blosclz', 'lz4', 'lz4hc', 'zlib', 'zstd'.
    - pixel_size (float): Pixel size in micrometers.
    - chunk_size (int): Number of TIFF images to load in each chunk.
    - verbose (bool): Enable verbose logging.

    Returns:
    - None
    """
    setup_custom_logger(verbose=verbose)
    
    dtype_map = {
        'int8': np.int8,
        'int16': np.int16,
        'int32': np.int32,
        'uint8': np.uint8,
        'uint16': np.uint16,
        'float32': np.float32,
        'float64': np.float64
    }
    
    global_min, global_max = minmaxHisto(input_dir)
    info(f"Global min and max found: {global_min}, {global_max}")
    
    start_index = 0
    mode = 'w'

    while True:
        stack, start_index = load_tiff_chunked(
            input_dir,
            dtype_map[dtype],
            chunk_size,
            start_index,
            global_min,
            global_max
        )
        if stack is None:
            break
        
        save_zarr(
            stack,
            output_path,
            chunks,
            compression,
            pixel_size,
            shard_shape,
            mode,
            original_dtype=dtype_map[dtype]
        )
        mode = 'a'
        
        if start_index >= len(os.listdir(input_dir)):
            break

if __name__ == "__main__":
    main()
