import click
import zarr
import tifffile as tiff
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from .log import info, error, setup_custom_logger

@click.command()
@click.argument('input_zarr', type=click.Path(exists=True, dir_okay=True, file_okay=False))
@click.argument('output_dir', type=click.Path(dir_okay=True, writable=True))
@click.option('--resolution', '-r', default='0', help="Resolution layer to convert (default is '0').")
@click.option('--dtype', default='float32', help="Data type for the output TIFF slices (default is 'float32').")
@click.option('--threads', default=4, help="Number of threads for parallel processing (default is 4).")
def convert_zarr_to_tiff(input_zarr, output_dir, resolution, dtype, threads):
    """
    Convert a specific resolution layer of a Zarr file to a stack of TIFF slices.

    INPUT_ZARR: Path to the input Zarr directory.
    OUTPUT_DIR: Path to the output directory where TIFF slices will be saved.
    """
    try:
        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            info(f"Created output directory: {output_dir}")

        # Open the Zarr file
        zarr_store = zarr.open(input_zarr, mode='r')
        info(f"Opened Zarr file: {input_zarr}")

        # Check if the requested resolution layer exists
        if resolution not in zarr_store:
            error(f"Resolution layer '{resolution}' not found in the Zarr file.")
            return

        # Access the specified resolution layer
        resolution_data = zarr_store[resolution]
        info(f"Accessed resolution layer: {resolution}")

        # Check if the resolution data is 3D
        if len(resolution_data.shape) != 3:
            error(f"Resolution layer '{resolution}' is not a 3D dataset.")
            return

        # Determine base name from Zarr prefix
        zarr_name = os.path.splitext(os.path.basename(input_zarr.rstrip('/')))[0]
        if not zarr_name:
            error("Zarr name could not be determined. Ensure the input path is correct.")
            return
        info(f"Zarr name determined: {zarr_name}")

        # Define a function to save a single slice
        def save_slice(i):
            slice_data = resolution_data[i, :, :]
            slice_converted = slice_data.astype(dtype)
            output_slice_path = os.path.join(output_dir, f"{zarr_name}_{i:04d}.tiff")
            tiff.imwrite(output_slice_path, slice_converted, photometric='minisblack')
            #info(f"Saved slice {i} to {output_slice_path}")
            return i

        # Process slices in parallel
        num_slices = resolution_data.shape[0]
        info(f"Starting conversion of {num_slices} slices using {threads} threads.")
        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = [executor.submit(save_slice, i) for i in range(num_slices)]

            # Use tqdm for progress tracking
            for _ in tqdm(as_completed(futures), total=num_slices, desc="Converting slices"):
                pass

        info(f"Successfully converted resolution layer '{resolution}' to TIFF slices in {output_dir}")
    except Exception as e:
        error(f"Error: {e}")

def main():
    setup_custom_logger(verbose=True)  # Initialize logger with verbose mode
    convert_zarr_to_tiff()


if __name__ == '__main__':
    main()
