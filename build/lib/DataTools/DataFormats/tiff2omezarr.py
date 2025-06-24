import os
import numpy as np
import zarr
from numcodecs import Blosc
import click
from ome_zarr.io import parse_url
from .utils import load_tiff_chunked, minmaxHisto
from .log import info, setup_custom_logger


def initialize_zarr_store(output_path, shape, dtype, chunks, compression):
    """
    Initializes the OME-Zarr store **only if it does not exist**.
    Otherwise, opens it for appending.
    """
    store = parse_url(output_path, mode='a').store
    group = zarr.open_group(store, mode='a')

    if "0" in group:
        info(f"Appending to existing Zarr store at {output_path}")
        return group  # Store already exists

    # First chunk: Create dataset
    info(f"Creating new Zarr store at {output_path}")
    compressor = Blosc(cname=compression, clevel=5, shuffle=2)

    group.create_dataset(
        name="0",
        shape=(0, shape[1], shape[2]),  # Start with Z=0, grows dynamically
        maxshape=(None, shape[1], shape[2]),  # Unlimited along Z-axis
        dtype=dtype,
        chunks=chunks,
        compressor=compressor,
        overwrite=True,
    )

    return group


def append_chunk_to_zarr(group, volume):
    """
    **Appends** a chunk to the existing Zarr dataset **without overwriting**.
    """
    dataset = group["0"]
    dataset.append(volume, axis=0)  # Append along Z-axis
    info(f"Appended {volume.shape} to Zarr")


@click.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--dtype', type=click.Choice(
    ['int8', 'int16', 'int32', 'uint8', 'uint16', 'float32', 'float64']),
    default='uint8', help='Data type of the images.'
)
@click.option('--pixel_size', type=float, default=1.0,
              help='Pixel size in micrometers.')
@click.option('--chunks', type=(int, int, int), default=(64, 64, 64),
              help='Chunk size for the Zarr array.')
@click.option('--compression',
              type=click.Choice(['blosclz', 'lz4', 'lz4hc', 'zlib', 'zstd']),
              default='lz4',
              help='Compression algorithm for Zarr storage.')
@click.option('--chunk_size', type=int, default=64,
              help='Number of TIFF images to load per chunk.')
@click.option('--verbose', is_flag=True, help='Enable verbose logging.')
def main(input_dir, output_path, dtype, pixel_size, chunks, compression, chunk_size, verbose):
    """
    Processes TIFF images in INPUT_DIR, loads in chunks, and **appends** to OME-Zarr store.
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
    group = None  # Will initialize only on first chunk

    while True:
        # Load a chunk
        stack, start_index = load_tiff_chunked(
            input_dir, dtype_map[dtype], chunk_size, start_index, global_min, global_max
        )
        if stack is None:
            break  # No more data

        # First chunk: Initialize Zarr store
        if group is None:
            shape = (0, stack.shape[1], stack.shape[2])
            group = initialize_zarr_store(output_path, shape, dtype_map[dtype], chunks, compression)

        # Append chunk to OME-Zarr store
        append_chunk_to_zarr(group, stack)

        info(f"Processed {start_index} slices.")

        if start_index >= len(os.listdir(input_dir)):
            break


if __name__ == "__main__":
    main()

