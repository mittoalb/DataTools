"""Convert a directory of TIFF slices into a Neuroglancer precomputed dataset.

Precomputed format:
    output_path/
        info                                    (JSON: scales, chunk sizes, dtype, encoding)
        <res_x>_<res_y>_<res_z>/                (mip 0 chunk directory, resolution in nm)
            <x0>-<x1>_<y0>-<y1>_<z0>-<z1>       (chunk file: raw / jpeg / zstd-raw / etc.)
            ...
        <res_x*2>_<res_y*2>_<res_z*2>/          (mip 1, 2x downsampled)
            ...

Axis convention: precomputed uses XYZ order throughout (resolution, chunk_size,
volume_size, slab writes). Our TIFFs load as (Z, Y, X), so we transpose to
(X, Y, Z) before every write — the cheap single-view transpose, no copy.

Encoding choices:
    raw                      - lossless, any dtype. Can be compressed with gzip/br/zstd.
    jpeg                     - lossy, 2D encoded per Z-slice, uint8 only. Smallest output.
    compressed_segmentation  - lossless, uint32/uint64 integer labels only. Not usable
                               for grayscale imagery.
"""
import os
import shutil
import sys

import click
import numpy as np
import tifffile
from cloudvolume import CloudVolume
from skimage.transform import downscale_local_mean
from tqdm import tqdm

from .log import info, setup_custom_logger
from .utils import load_tiff_chunked
from .tiff2zarr import (
    calculate_percentile_limits,
    calculate_histogram_based_limits,
    scale_to_dtype,
)


# ---------------------------------------------------------------------------
# Encoding / dtype compatibility matrix
# ---------------------------------------------------------------------------

# What (encoding, dtype) combinations are legal in precomputed.
_LEGAL = {
    ('raw', 'uint8'), ('raw', 'uint16'), ('raw', 'uint32'), ('raw', 'uint64'),
    ('raw', 'int8'),  ('raw', 'int16'),  ('raw', 'int32'),  ('raw', 'int64'),
    ('raw', 'float32'),
    ('jpeg', 'uint8'),
    ('compressed_segmentation', 'uint32'),
    ('compressed_segmentation', 'uint64'),
}


def _validate_encoding(encoding, dtype):
    if (encoding, dtype) not in _LEGAL:
        legal_for_enc = sorted({d for e, d in _LEGAL if e == encoding})
        raise click.BadParameter(
            f"--encoding {encoding} is incompatible with --dtype {dtype}. "
            f"Legal dtypes for {encoding}: {legal_for_enc}"
        )


# ---------------------------------------------------------------------------
# Scale planning
# ---------------------------------------------------------------------------

def _auto_max_mip(volume_size_xyz, chunk_xyz):
    """Pick the deepest pyramid that still has at least one chunk per axis.

    Precomputed convention: mip L has size ceil(size / 2**L) per axis.
    We stop when any axis would drop below its chunk size at the NEXT level.
    """
    L = 0
    while True:
        next_L = L + 1
        shape_at_next = [max(1, s // (2 ** next_L)) for s in volume_size_xyz]
        if any(s < c for s, c in zip(shape_at_next, chunk_xyz)):
            return L
        L = next_L
        if L >= 10:   # safety cap; nobody needs 11 mips
            return L


# ---------------------------------------------------------------------------
# Pyramid downsampling (inline, per chunk-slab)
# ---------------------------------------------------------------------------

def _downsample_slab(slab_zyx, mip):
    """Downsample a (Z,Y,X) slab by factor 2**mip in all 3 axes.

    Factor 1 (mip 0) returns the slab as-is. Uses successive block-mean passes
    so the result matches CloudVolume's default downsample arithmetic.
    """
    out = slab_zyx
    for _ in range(mip):
        if any(s < 2 for s in out.shape):
            return None
        out = downscale_local_mean(out, (2, 2, 2)).astype(slab_zyx.dtype)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--dtype', type=click.Choice(['uint8', 'uint16', 'uint32', 'uint64',
                                            'int8', 'int16', 'int32', 'int64',
                                            'float32']),
              default='uint8', show_default=True,
              help='Output voxel dtype. Must be compatible with --encoding.')
@click.option('--encoding', type=click.Choice(['raw', 'jpeg', 'compressed_segmentation']),
              default='raw', show_default=True,
              help='Precomputed chunk encoding.')
@click.option('--compress', type=click.Choice(['gzip', 'br', 'zstd', 'none']),
              default='gzip', show_default=True,
              help='Transport compression applied over raw chunks '
                   '(ignored for jpeg; cloud-volume defaults to gzip for raw).')
@click.option('--jpeg-quality', type=click.IntRange(1, 100), default=85, show_default=True,
              help='JPEG quality (--encoding jpeg only).')
@click.option('--chunks', type=(int, int, int), default=(64, 64, 64), show_default=True,
              help='Chunk shape in XYZ voxels. 64 64 64 is the Neuroglancer default.')
@click.option('--resolution', type=(int, int, int), default=(1000, 1000, 1000),
              show_default=True,
              help='Voxel resolution in nanometers, XYZ order. Default = 1 micrometer isotropic.')
@click.option('--max-mip', type=click.IntRange(min=0), default=0, show_default=True,
              help='Number of pyramid levels above mip 0. 0 = auto-pick from input dims.')
@click.option('--chunk_size', type=int, default=64, show_default=True,
              help='Number of TIFF slices loaded per iteration (RAM budget).')
@click.option('--min_val', type=float, default=None,
              help='Manual min for intensity scaling. If given with --max_val, '
                   'the percentile scan is skipped.')
@click.option('--max_val', type=float, default=None, help='Manual max for intensity scaling.')
@click.option('--min_percentile', type=float, default=0.5, show_default=True)
@click.option('--max_percentile', type=float, default=99.5, show_default=True)
@click.option('--sample_ratio', type=int, default=100, show_default=True,
              help='Pixel sub-sampling for percentile scan.')
@click.option('--use_histogram', is_flag=True,
              help='Use histogram method instead of percentile scan.')
@click.option('--layer-type', type=click.Choice(['image', 'segmentation']),
              default='image', show_default=True)
@click.option('--parallel', type=click.IntRange(min=1), default=1, show_default=True,
              help='CloudVolume parallel write processes.')
@click.option('--verbose', is_flag=True)
def main(input_dir, output_path, dtype, encoding, compress, jpeg_quality, chunks,
         resolution, max_mip, chunk_size, min_val, max_val,
         min_percentile, max_percentile, sample_ratio, use_histogram,
         layer_type, parallel, verbose):
    """Convert a directory of TIFF slices to a Neuroglancer precomputed dataset.

    INPUT_DIR  directory of *.tif / *.tiff slices, sorted lexicographically along Z
    OUTPUT_PATH  destination (local path; wrapped as file:// for CloudVolume)
    """
    setup_custom_logger(verbose=verbose)
    _validate_encoding(encoding, dtype)
    if encoding == 'jpeg':
        compress = None  # jpeg is its own compression

    dtype_map = {
        'int8': np.int8, 'int16': np.int16, 'int32': np.int32, 'int64': np.int64,
        'uint8': np.uint8, 'uint16': np.uint16, 'uint32': np.uint32, 'uint64': np.uint64,
        'float32': np.float32,
    }
    target_dtype = dtype_map[dtype]

    # --- 1. Discover input dims ---
    tiff_files = sorted(
        f for f in os.listdir(input_dir)
        if f.lower().endswith(('.tif', '.tiff'))
    )
    if not tiff_files:
        raise click.ClickException(f"No TIFF files in {input_dir}")
    total_z = len(tiff_files)
    sample = tifffile.imread(os.path.join(input_dir, tiff_files[0]))
    real_y, real_x = sample.shape[-2:]
    volume_size_xyz = (real_x, real_y, total_z)  # precomputed is XYZ
    info(f"Input: {total_z} TIFFs, each {real_y} x {real_x}  (volume XYZ = {volume_size_xyz})")

    # --- 2. Intensity scaling range ---
    if min_val is not None and max_val is not None:
        if min_val >= max_val:
            raise click.BadParameter(f"min_val ({min_val}) must be < max_val ({max_val})")
        data_min, data_max = min_val, max_val
        info(f"Manual intensity range: [{data_min}, {data_max}]")
    elif min_val is not None or max_val is not None:
        raise click.BadParameter("--min_val and --max_val must be given together or not at all")
    else:
        if use_histogram:
            data_min, data_max = calculate_histogram_based_limits(input_dir)
        else:
            data_min, data_max = calculate_percentile_limits(
                input_dir, min_percentile, max_percentile, sample_ratio
            )

    # --- 3. Pyramid depth ---
    if max_mip == 0:
        max_mip = _auto_max_mip(volume_size_xyz, chunks)
        info(f"Auto-selected max_mip = {max_mip} (pyramid has mip 0..{max_mip})")

    # --- 4. Build info and create volume ---
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    cv_path = f"file://{os.path.abspath(output_path)}"

    cv_info = CloudVolume.create_new_info(
        num_channels=1,
        layer_type=layer_type,
        data_type=dtype,
        encoding=encoding,
        resolution=list(resolution),           # XYZ, nm
        voxel_offset=[0, 0, 0],
        chunk_size=list(chunks),               # XYZ
        volume_size=list(volume_size_xyz),     # XYZ
        max_mip=max_mip,
        factor=(2, 2, 2),
    )
    if encoding == 'jpeg':
        # Record quality in info so readers (and we) know.
        for scale in cv_info['scales']:
            scale['jpeg_quality'] = jpeg_quality

    # fill_missing=True: on non-aligned writes CloudVolume reads the existing
    # chunk first; a chunk that doesn't exist yet should be treated as zeros,
    # not raise EmptyVolumeException. non_aligned_writes allows the partial
    # slab write at all; fill_missing makes the read half of the read-modify-
    # write cycle succeed.
    vol_mip0 = CloudVolume(cv_path, info=cv_info, mip=0, compress=compress,
                           progress=False, parallel=parallel,
                           non_aligned_writes=True, fill_missing=True)
    vol_mip0.commit_info()
    info(f"=== Writing Neuroglancer precomputed ===")
    info(f"  path:        {cv_path}")
    info(f"  encoding:    {encoding}" + (f" (jpeg_quality={jpeg_quality})" if encoding == 'jpeg' else ""))
    info(f"  compress:    {compress}")
    info(f"  dtype:       {dtype}")
    info(f"  resolution:  {resolution} nm (XYZ)")
    info(f"  chunk_size:  {chunks} voxels (XYZ)")
    info(f"  volume_size: {volume_size_xyz} voxels (XYZ)")
    info(f"  mip levels:  0..{max_mip}")

    # Pre-create volume handles for each mip > 0 so we can write their slabs
    # as we go.
    vols_by_mip = {0: vol_mip0}
    for L in range(1, max_mip + 1):
        vols_by_mip[L] = CloudVolume(cv_path, mip=L, compress=compress,
                                     progress=False, parallel=parallel,
                                     non_aligned_writes=True, fill_missing=True)

    # --- 5. Chunked streaming write ---
    start_index = 0
    estimated_chunks = (total_z + chunk_size - 1) // chunk_size
    info(f"Processing {total_z} TIFFs in {estimated_chunks} load-chunks of up to {chunk_size}")

    with tqdm(total=estimated_chunks, desc="Chunks", unit="chunk") as pbar:
        while True:
            # Loader returns (chunk_zyx, next_start)
            stack, start_index = load_tiff_chunked(
                input_dir, np.float32, chunk_size, start_index
            )
            if stack is None:
                break
            cz = stack.shape[0]
            z_offset = start_index - cz
            info(f"  slab z=[{z_offset}, {z_offset + cz}), shape={stack.shape}")

            scaled = scale_to_dtype(stack, data_min, data_max, target_dtype)

            # Write mip 0 and every higher mip whose block boundary this slab aligns to.
            for L, vol in vols_by_mip.items():
                down = _downsample_slab(scaled, L)
                if down is None:
                    continue
                # z range at this mip
                z0 = z_offset // (2 ** L)
                z1 = z0 + down.shape[0]
                # Clamp to volume_size at this mip (last slab's tail may overshoot by 1)
                mip_vol_z = vol.volume_size[2]
                z1 = min(z1, mip_vol_z)
                down = down[: z1 - z0]
                # transpose (Z, Y, X) -> (X, Y, Z) for cloud-volume
                vol[0:real_x >> L if L else real_x,
                    0:real_y >> L if L else real_y,
                    z0:z1] = down.transpose(2, 1, 0)

            pbar.update(1)
            if start_index >= total_z:
                break

    info(f"=== Done. Output: {output_path} ===")
    info(f"Open in Neuroglancer with:  precomputed://file://{os.path.abspath(output_path)}")
    info(f"Or over HTTP:               precomputed://http://<host>:<port>/{os.path.basename(output_path)}")


if __name__ == '__main__':
    main()
