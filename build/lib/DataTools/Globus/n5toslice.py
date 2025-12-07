#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 14:15:40 2025

@author: amittone
"""

#!/usr/bin/env python3

import click
import z5py
from PIL import Image
import matplotlib.pyplot as plt

@click.command()
@click.option(
    "--n5-file-path", "-f",
    required=True,
    type=str,
    help="Path to your N5 file (e.g. 'P-1C_290_rec.n5')."
)
@click.option(
    "--resolution", "-r",
    default="0",
    show_default=True,
    help="Dataset key inside the N5 file (the 'internal' dataset name)."
)
@click.option(
    "--z-index", "-z",
    default=50,
    show_default=True,
    type=int,
    help="Z-slice index from the 3D volume to extract."
)
@click.option(
    "--output", "-o",
    default="N5Slice.tiff",
    show_default=True,
    help="Output file name for the 2D slice."
)
@click.option(
    "--no-show",
    is_flag=True,
    default=False,
    help="If set, don't open a matplotlib window to display the slice."
)
def main(n5_file_path, dataset_key, z_index, output, no_show):
    """
    Reads a 3D volume from an N5 file and saves a single Z-slice as an image.
    Optionally, display the slice with matplotlib.
    """

    # Open the N5 file in read-only mode
    f = z5py.File(n5_file_path, mode='r', use_zarr_format=False)

    # Access the 3D dataset inside the N5
    ds = f[dataset_key]

    print(f"Dataset shape: {ds.shape}  (e.g., (Z, Y, X))")
    print(f"Dataset dtype: {ds.dtype}")

    # Read the slice you want
    slice_2d = ds[z_index, :, :]   # shape will be (Y, X)

    # Save the slice as an image (TIFF)
    img = Image.fromarray(slice_2d)
    img.save(output)
    print(f"Saved slice {z_index} to '{output}'")

    # Show the slice with matplotlib (unless --no-show)
    if not no_show:
        plt.figure()
        plt.imshow(slice_2d, cmap='gray')
        plt.title(f"Z-Slice {z_index}")
        plt.axis("off")
        plt.show()

if __name__ == "__main__":

   """
   Example usage:
	python N5toslice.py \
	  --n5-file-path P-1C_290_rec.n5 \
	  --dataset-key 0 \
	  --z-index 50 \
	  --output slice_50.tiff
   """
   main()
