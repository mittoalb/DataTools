import os
import shutil

import click
import numpy as np
import zarr
from numcodecs import Blosc
from PIL import Image
from skimage.transform import downscale_local_mean

ZARR_MAJOR = int(zarr.__version__.split('.')[0])


# ---------------------------------------------------------------------------
# Zarr v2/v3 compatibility helpers (same pattern as tiff2zarr.py)
# ---------------------------------------------------------------------------

def _open_group(output_path, mode, zarr_format):
    """Open a Zarr group of the requested format.

    zarr_format=2 works on both zarr 2.x and 3.x. zarr_format=3 requires zarr>=3.
    """
    if zarr_format == 3 and ZARR_MAJOR < 3:
        raise RuntimeError(
            f"--zarr-format 3 requires zarr>=3; installed is {zarr.__version__}"
        )
    if ZARR_MAJOR >= 3:
        return zarr.open_group(output_path, mode=mode, zarr_format=zarr_format)
    return zarr.open_group(output_path, mode=mode)


def _make_compressor(compression, zarr_format):
    if zarr_format == 3:
        from zarr.codecs import BloscCodec
        return BloscCodec(cname=compression, clevel=5, shuffle='bitshuffle')
    return Blosc(cname=compression, clevel=5, shuffle=2)


def _create_array(group, name, shape, chunks, dtype, compressor):
    if ZARR_MAJOR >= 3:
        return group.create_array(
            name=name, shape=shape, chunks=chunks, dtype=dtype,
            compressors=[compressor],
        )
    return group.create_dataset(
        name=name, shape=shape, chunks=chunks, dtype=dtype, compressor=compressor
    )


# ---------------------------------------------------------------------------
# BMP loading & cropping (unchanged logic)
# ---------------------------------------------------------------------------

def compute_auto_crop(input_dir, dtype=np.uint8, threshold=10, pad=0):
    files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith('.bmp'))
    if not files:
        raise ValueError("No BMP files found")

    max_proj = None
    ref_shape = None

    for f in files:
        img = Image.open(os.path.join(input_dir, f)).convert("L")
        img_np = np.asarray(img, dtype=dtype)

        if ref_shape is None:
            ref_shape = img_np.shape
            max_proj = img_np.copy()
        else:
            if img_np.shape != ref_shape:
                continue
            max_proj = np.maximum(max_proj, img_np)

    mask = max_proj > threshold
    nonzero = np.argwhere(mask)
    if nonzero.size == 0:
        raise RuntimeError("Autocrop failed: no pixels > threshold")

    min_y, min_x = nonzero.min(axis=0)
    max_y, max_x = nonzero.max(axis=0) + 1

    min_x = max(min_x - pad, 0)
    min_y = max(min_y - pad, 0)
    max_x = min(max_x + pad, ref_shape[1])
    max_y = min(max_y + pad, ref_shape[0])

    return (min_x, max_x, min_y, max_y)


def load_bmp_stack(input_dir, dtype, crop_box=None):
    files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith('.bmp'))
    if not files:
        raise ValueError("No BMP files found in the input directory.")

    images = []
    target_shape = None
    skipped = 0

    for f in files:
        img_path = os.path.join(input_dir, f)
        img = Image.open(img_path).convert("L")

        if crop_box:
            startx, endx, starty, endy = crop_box
            img = img.crop((startx, starty, endx, endy))

        img_np = np.asarray(img, dtype=dtype)

        if target_shape is None:
            target_shape = img_np.shape
        elif img_np.shape != target_shape:
            print(f"Skipping {f} (cropped shape {img_np.shape} != {target_shape})")
            skipped += 1
            continue

        images.append(img_np)

    if not images:
        raise RuntimeError("No consistent BMP images found after cropping.")

    stack = np.stack(images, axis=0)
    print(f"Loaded volume shape: {stack.shape}")
    if skipped > 0:
        print(f"{skipped} image(s) were skipped due to inconsistent shape.")
    return stack


# ---------------------------------------------------------------------------
# Pyramid construction
# ---------------------------------------------------------------------------

def calculate_levels(shape, mode='2d'):
    """Max N such that the relevant dims are divisible by 2**N.

    mode='2d' considers only Y/X; mode='3d' considers Z/Y/X.
    """
    get_divisions = lambda n: (n & -n).bit_length() - 1 if n > 0 else 0
    dims = shape[1:] if mode == '2d' else shape
    return min(get_divisions(s) for s in dims)


def downsample(volume, levels, mode='2d'):
    pyramid = [volume]
    factors = (1, 2, 2) if mode == '2d' else (2, 2, 2)
    for level in range(1, levels + 1):
        current = pyramid[-1]
        active = current.shape[1:] if mode == '2d' else current.shape
        if min(active) < 2:
            print(f"Stopping at level {level}: dims too small")
            break
        down = downscale_local_mean(current, factors)
        pyramid.append(down)
        print(f"Level {level}: {down.shape}")
    return pyramid


# ---------------------------------------------------------------------------
# Zarr writer
# ---------------------------------------------------------------------------

def save_zarr(volume, output_path, chunks, compression, pixel_size,
              original_dtype=np.uint8, downsample_mode='2d', zarr_format=2):
    """Write a BMP-derived volume as a multiscale Zarr store (v2 or v3)."""
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    root = _open_group(output_path, mode='w', zarr_format=zarr_format)
    root.attrs["zarr_format"] = int(zarr_format)
    compressor = _make_compressor(compression, zarr_format)

    target_levels = calculate_levels(volume.shape, mode=downsample_mode)
    root.attrs["target_levels"] = int(target_levels)
    print(f"Calculated {target_levels} pyramid levels for mode: {downsample_mode}")

    pyramid = downsample(volume, target_levels, mode=downsample_mode)
    datasets = []

    for level, data in enumerate(pyramid):
        data = data.astype(original_dtype)
        name = f"{level}"
        level_chunks = tuple(min(c, s) for c, s in zip(chunks, data.shape))
        z = _create_array(root, name, data.shape, level_chunks, data.dtype, compressor)
        z[:] = data

        if downsample_mode == '2d':
            scale = [1.0, 2 ** level, 2 ** level]
            # Only Y/X are downsampled; Z stays voxel-aligned.
            offset = [0.0, (2 ** level - 1) / 2.0, (2 ** level - 1) / 2.0]
        else:
            scale = [2 ** level] * 3
            offset = [(2 ** level - 1) / 2.0] * 3

        datasets.append({
            "path": name,
            "coordinateTransformations": [
                {"type": "scale", "scale": [pixel_size * s for s in scale]},
                {"type": "translation", "translation": [pixel_size * o for o in offset]},
            ],
        })

    # NGFF spec version is pinned to the zarr format (Neuroglancer's v3 reader
    # rejects root-level multiscales as invalid; v3 wants the "ome" key).
    ngff_version = "0.5" if zarr_format == 3 else "0.4"
    multiscales = [{
        "version": ngff_version,
        "name": "bmp_stack",
        "axes": [
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ],
        "datasets": datasets,
        "type": "mean",
        "metadata": {
            "method": "skimage.transform.downscale_local_mean",
            "mode": downsample_mode,
        },
    }]
    if zarr_format == 3:
        root.attrs["ome"] = {"version": ngff_version, "multiscales": multiscales}
    else:
        root.attrs["multiscales"] = multiscales
    root.attrs["pixel_size"] = pixel_size
    print(f"Zarr v{zarr_format} store saved at: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--dtype', type=click.Choice(['int8', 'int16', 'int32', 'uint8', 'uint16', 'float32', 'float64']),
              default='uint8', help='Data type for output volume.')
@click.option('--chunks', type=(int, int, int), default=(64, 64, 64), help='Zarr chunk size (Z, Y, X).')
@click.option('--compression', type=click.Choice(['blosclz', 'lz4', 'lz4hc', 'zlib', 'zstd']),
              default='blosclz', help='Compression algorithm.')
@click.option('--pixel_size', type=float, default=1.0, help='Pixel size in micrometers.')
@click.option('--crop', type=str, default=None, help='2D crop in format: startx:endx:starty:endy')
@click.option('--autocrop', is_flag=True, help='Automatically crop to non-zero region using thresholded max projection.')
@click.option('--autocrop-threshold', type=int, default=10, help='Pixel intensity threshold for autocrop.')
@click.option('--pad', type=int, default=0, help='Padding to apply around the autocrop region.')
@click.option('--downsample-mode', type=click.Choice(['2d', '3d']), default='2d',
              help='Use 2d (Y/X only) or full 3d (Z/Y/X) downsampling.')
@click.option('--zarr-format', 'zarr_format', type=click.Choice(['2', '3']), default='2',
              show_default=True, help='Zarr on-disk format to write. v3 requires zarr>=3.')
def main(input_dir, output_path, dtype, chunks, compression, pixel_size,
         crop, autocrop, autocrop_threshold, pad, downsample_mode, zarr_format):
    zarr_format = int(zarr_format)
    print(f"=== Writing Zarr v{zarr_format} (library zarr=={zarr.__version__}) ===")

    dtype_map = {
        'int8': np.int8, 'int16': np.int16, 'int32': np.int32,
        'uint8': np.uint8, 'uint16': np.uint16,
        'float32': np.float32, 'float64': np.float64
    }

    crop_box = None
    if autocrop:
        crop_box = compute_auto_crop(input_dir, dtype_map[dtype],
                                     threshold=autocrop_threshold,
                                     pad=pad)
        print(f"Autocrop box: {crop_box}")
    elif crop:
        try:
            parts = list(map(int, crop.split(':')))
            assert len(parts) == 4
            crop_box = (parts[0], parts[1], parts[2], parts[3])
        except Exception:
            raise ValueError("Crop format must be: startx:endx:starty:endy")

    volume = load_bmp_stack(input_dir, dtype_map[dtype], crop_box=crop_box)
    save_zarr(volume, output_path, chunks, compression, pixel_size,
              original_dtype=dtype_map[dtype],
              downsample_mode=downsample_mode,
              zarr_format=zarr_format)
    print(f"=== Wrote Zarr v{zarr_format} to {output_path} ===")


if __name__ == "__main__":
    main()
