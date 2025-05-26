import os
import shutil
import numpy as np
import zarr
import click
from PIL import Image
from numcodecs import Blosc
from skimage.transform import downscale_local_mean


def load_bmp_stack(input_dir, dtype, crop_box=None):
    files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith('.bmp'))
    if not files:
        raise ValueError("No BMP files found in the input directory.")

    images = []
    target_shape = None
    for f in files:
        img_path = os.path.join(input_dir, f)
        img = Image.open(img_path).convert("L")  # grayscale

        if crop_box:
            startx, endx, starty, endy = crop_box
            img = img.crop((startx, starty, endx, endy))  # crop before checking shape

        img_np = np.asarray(img, dtype=dtype)

        if target_shape is None:
            target_shape = img_np.shape
        elif img_np.shape != target_shape:
            print(f"Skipping {f} (cropped shape {img_np.shape} ≠ {target_shape})")
            continue

        images.append(img_np)

    if not images:
        raise RuntimeError("No consistent BMP images found after cropping. Check input.")

    stack = np.stack(images, axis=0)  # Shape: (Z, Y, X)
    print(f"Loaded volume shape: {stack.shape}")
    return stack


def calculate_levels(data):
    min_dim = min(data.shape)
    levels = 0
    while min_dim >= 2:
        min_dim //= 2
        levels += 1
    return levels


def downsample(volume, levels, mode='2d'):
    pyramid = [volume]
    for level in range(1, levels + 1):
        current = pyramid[-1]
        if mode == '2d':
            if current.shape[1] < 2 or current.shape[2] < 2:
                print(f"⚠️ Stopping at level {level}: Y/X too small")
                break
            factors = (1, 2, 2)
        else:  # 3d
            if min(current.shape) < 2:
                print(f"⚠️ Stopping at level {level}: Z/Y/X too small")
                break
            factors = (2, 2, 2)

        down = downscale_local_mean(current, factors)
        pyramid.append(down)
        print(f"Level {level}: {down.shape}")
    return pyramid


def save_zarr(volume, output_path, chunks, compression, pixel_size,
              original_dtype=np.uint8, downsample_mode='2d'):
    store = zarr.DirectoryStore(output_path)
    compressor = Blosc(cname=compression, clevel=5, shuffle=2)

    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    root = zarr.group(store=store)

    levels = min(calculate_levels(volume), 6)
    print(f"Calculated {levels} pyramid levels")

    pyramid = downsample(volume, levels, mode=downsample_mode)
    datasets = []

    for level, data in enumerate(pyramid):
        data = data.astype(original_dtype)
        name = f"{level}"
        z = root.create_dataset(name, shape=data.shape, chunks=chunks, dtype=data.dtype, compressor=compressor)
        z[:] = data

        scale = 2 ** level
        datasets.append({
            "path": name,
            "coordinateTransformations": [
                {"type": "scale", "scale": [scale, scale, scale]},
                {"type": "translation", "translation": [scale / 2 - 0.5] * 3}
            ]
        })

    root.attrs["multiscales"] = [{
        "version": "0.4",
        "name": "bmp_stack",
        "axes": [
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"}
        ],
        "datasets": datasets,
        "type": "gaussian"
    }]
    root.attrs["pixel_size"] = pixel_size
    print(f"Zarr store saved at: {output_path}")


@click.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--dtype', type=click.Choice(['int8', 'int16', 'int32', 'uint8', 'uint16', 'float32', 'float64']),
              default='uint8', help='Data type for output volume.')
@click.option('--chunks', type=(int, int, int), default=(64, 64, 64), help='Zarr chunk size (Z, Y, X).')
@click.option('--compression', type=click.Choice(['blosclz', 'lz4', 'lz4hc', 'zlib', 'zstd']),
              default='blosclz', help='Compression algorithm.')
@click.option('--pixel_size', type=float, default=1.0, help='Pixel size in micrometers.')
@click.option('--crop', type=str, default=None,
              help='2D crop in format: startx:endx:starty:endy')
@click.option('--downsample-mode', type=click.Choice(['2d', '3d']), default='2d',
              help='Use 2d (Y/X only) or full 3d (Z/Y/X) downsampling.')
def main(input_dir, output_path, dtype, chunks, compression, pixel_size, crop, downsample_mode):
    dtype_map = {
        'int8': np.int8, 'int16': np.int16, 'int32': np.int32,
        'uint8': np.uint8, 'uint16': np.uint16,
        'float32': np.float32, 'float64': np.float64
    }

    crop_box = None
    if crop:
        try:
            parts = list(map(int, crop.split(':')))
            assert len(parts) == 4
            crop_box = (parts[0], parts[1], parts[2], parts[3])
        except:
            raise ValueError("Crop format must be: startx:endx:starty:endy")

    volume = load_bmp_stack(input_dir, dtype_map[dtype], crop_box=crop_box)
    save_zarr(volume, output_path, chunks, compression, pixel_size,
              original_dtype=dtype_map[dtype], downsample_mode=downsample_mode)


if __name__ == "__main__":
    main()
