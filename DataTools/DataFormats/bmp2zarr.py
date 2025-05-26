import os
import shutil
import numpy as np
import zarr
import click
from PIL import Image
from numcodecs import Blosc
from skimage.transform import downscale_local_mean


def load_bmp_chunked(input_dir, dtype, chunk_size, start_index, min_val=0, max_val=255):
    files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith('.bmp'))
    if start_index >= len(files):
        return None, start_index

    end_index = min(start_index + chunk_size, len(files))
    images = []
    for f in files[start_index:end_index]:
        img = Image.open(os.path.join(input_dir, f))
        img = np.asarray(img, dtype=dtype)
        images.append(img)

    stack = np.stack(images, axis=0)
    return stack, end_index


def downsample(volume, levels):
    pyramid = [volume]
    for _ in range(levels):
        current = pyramid[-1]
        if min(current.shape) < 2:
            break
        current = downscale_local_mean(current, (1, 2, 2))
        pyramid.append(current)
    return pyramid


def calculate_levels(data):
    get_divisions = lambda n: (n & -n).bit_length() - 1
    dim = [get_divisions(s) for s in data.shape]
    return min(dim)


def save_zarr(volume, output_path, chunks, compression, pixel_size, mode='w', original_dtype=np.uint8):
    store = zarr.DirectoryStore(output_path)
    compressor = Blosc(cname=compression, clevel=5, shuffle=2)

    if mode == 'w' and os.path.exists(output_path):
        shutil.rmtree(output_path)

    root_group = zarr.open(store=store, mode=mode)
    levels = min(calculate_levels(volume), 6)
    pyramid_levels = downsample(volume, levels)
    datasets = []

    for level, data in enumerate(pyramid_levels):
        data = data.astype(original_dtype)
        name = f"{level}"
        z = root_group.create_dataset(name, shape=data.shape, chunks=chunks, dtype=data.dtype, compressor=compressor)
        z[:] = data

        scale = 2 ** level
        datasets.append({
            "path": name,
            "coordinateTransformations": [
                {"type": "scale", "scale": [scale]*3},
                {"type": "translation", "translation": [scale/2 - 0.5]*3}
            ]
        })

    root_group.attrs["multiscales"] = [{
        "version": "0.4",
        "name": "bmp_stack",
        "axes": [{"name": ax, "type": "space", "unit": "micrometer"} for ax in "zyx"],
        "datasets": datasets,
        "type": "gaussian"
    }]


@click.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--dtype', type=click.Choice(['int8', 'int16', 'int32', 'uint8', 'uint16', 'float32', 'float64']), default='uint8')
@click.option('--chunks', type=(int, int, int), default=(64, 64, 64))
@click.option('--compression', type=click.Choice(['blosclz', 'lz4', 'lz4hc', 'zlib', 'zstd']), default='blosclz')
@click.option('--pixel_size', type=float, default=1.0)
@click.option('--chunk_size', type=int, default=64)
def main(input_dir, output_path, dtype, chunks, compression, pixel_size, chunk_size):
    dtype_map = {'int8': np.int8, 'int16': np.int16, 'int32': np.int32, 'uint8': np.uint8,
                 'uint16': np.uint16, 'float32': np.float32, 'float64': np.float64}

    start_index = 0
    mode = 'w'
    while True:
        stack, start_index = load_bmp_chunked(input_dir, dtype_map[dtype], chunk_size, start_index)
        if stack is None:
            break
        save_zarr(stack, output_path, chunks, compression, pixel_size, mode, original_dtype=dtype_map[dtype])
        mode = 'a'
        if start_index >= len(os.listdir(input_dir)):
            break


if __name__ == "__main__":
    main()
