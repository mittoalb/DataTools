"""Convert a 3D dataset inside an HDF5 file into a multiscale OME-Zarr store.

Same pipeline as tiff2zarr (intensity limits -> optional per-slice
normalization -> scale_to_dtype -> padded pyramid -> cropped write), but the
Z slices come from an HDF5 dataset instead of a folder of TIFFs. The Zarr
writer, pyramid and metadata code is imported from tiff2zarr so both tools
produce identical stores.
"""
import click
import h5py
import numpy as np
import zarr
from tqdm import tqdm

try:  # registers Blosc/bitshuffle/LZ4/... HDF5 filters if the file uses them
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

from .log import info, setup_custom_logger
from .tiff2zarr import (
    ZARR_MAJOR,
    _combine_stats,
    _make_slice_stat_fn,
    calculate_levels,
    pad_for_pyramid,
    save_zarr,
    scale_to_dtype,
)

DEFAULT_DATASETS = ('/exchange/data',)


# ---------------------------------------------------------------------------
# Dataset selection / iteration
# ---------------------------------------------------------------------------

def find_dataset(h5, dataset_path=None):
    """Return the 3D dataset to convert.

    If dataset_path is given, use it. Otherwise: a well-known path
    (/exchange/data) if present, else the only 3D numeric dataset in the file.
    """
    if dataset_path:
        if dataset_path not in h5:
            raise click.BadParameter(f"Dataset {dataset_path!r} not found in file")
        dset = h5[dataset_path]
        if not isinstance(dset, h5py.Dataset):
            raise click.BadParameter(f"{dataset_path!r} is a group, not a dataset")
        return dset

    candidates = []

    def _visit(name, obj):
        if isinstance(obj, h5py.Dataset) and obj.ndim == 3 and obj.dtype.kind in 'iuf':
            candidates.append('/' + name)

    h5.visititems(_visit)

    for path in DEFAULT_DATASETS:
        if path in candidates:
            return h5[path]
    if len(candidates) == 1:
        return h5[candidates[0]]
    if not candidates:
        raise click.ClickException("No 3D numeric dataset found in the HDF5 file")
    listing = "\n".join(f"  {p}  shape={h5[p].shape} dtype={h5[p].dtype}" for p in candidates)
    raise click.ClickException(
        f"Several 3D datasets found; choose one with --dataset:\n{listing}"
    )


def iter_blocks(dset, z0, z1, block):
    """Yield (offset_from_z0, float32 array) blocks of up to `block` Z slices."""
    for s in range(z0, z1, block):
        e = min(s + block, z1)
        yield s - z0, dset[s:e].astype(np.float32)


# ---------------------------------------------------------------------------
# Intensity statistics (HDF5 versions of the tiff2zarr helpers)
# ---------------------------------------------------------------------------

def compute_slice_scales(dset, z0, z1, block, stat_spec, reference_how):
    """Per-slice multiplicative factors that flatten slice-to-slice drift."""
    stat_fn = _make_slice_stat_fn(stat_spec)
    n = z1 - z0
    info(f"Normalization pre-pass: computing per-slice {stat_spec} for {n} slices...")
    stats = np.empty(n, dtype=np.float64)
    with tqdm(total=n, desc=f"Slice {stat_spec}", unit="slices") as pbar:
        for off, arr in iter_blocks(dset, z0, z1, block):
            for i, img in enumerate(arr):
                stats[off + i] = stat_fn(img)
            pbar.update(arr.shape[0])
    reference = _combine_stats(stats, reference_how)
    if reference <= 0:
        raise ValueError(f"Reference ({reference_how} of per-slice {stat_spec}) is {reference}; cannot normalize")
    nonpos = int((stats <= 0).sum())
    scales = np.where(stats > 0, reference / stats, 1.0)
    spread_pct = (stats.max() - stats.min()) / reference * 100
    info(f"Per-slice {stat_spec}: min={stats.min():.4f} max={stats.max():.4f} "
         f"spread={spread_pct:.1f}% of {reference_how} reference ({reference:.4f})")
    info(f"Scale factors: min={scales.min():.4f} max={scales.max():.4f}")
    if nonpos:
        info(f"Warning: {nonpos} slices had {stat_spec} <= 0; left unscaled")
    return scales


def sample_volume(dset, z0, z1, n_slices=256, max_samples=20_000_000, scales=None):
    """Sample voxels for automatic intensity limits without reading the whole volume.

    Reads `n_slices` Z slices spread evenly over [z0, z1) (0 = every slice) and
    keeps a regular 2D grid of pixels from each, with the grid stride chosen so
    the total stays under `max_samples` (~80 MB of float32 at the default).
    Returns a 1D float32 array; per-slice normalization `scales` (indexed from
    z0) is applied so the limits match what scale_to_dtype will see.
    """
    n = z1 - z0
    if n_slices <= 0 or n_slices >= n:
        z_idx = np.arange(z0, z1)
    else:
        z_idx = np.unique(np.linspace(z0, z1 - 1, n_slices).round().astype(np.int64))
    ny, nx = dset.shape[1:]
    per_slice = max(1, max_samples // len(z_idx))
    stride = max(1, int(np.ceil(np.sqrt(ny * nx / per_slice))))
    info(f"Sampling {len(z_idx)} of {n} slices, every {stride} px in Y and X "
         f"(~{len(z_idx) * ((ny + stride - 1) // stride) * ((nx + stride - 1) // stride):,} samples)")

    samples = []
    with tqdm(total=len(z_idx), desc="Sampling slices", unit="slices") as pbar:
        for z in z_idx:
            img = dset[int(z)][::stride, ::stride].astype(np.float32)
            if scales is not None:
                img *= np.float32(scales[z - z0])
            samples.append(img.ravel())
            pbar.update(1)
    return np.concatenate(samples)


def calculate_percentile_limits(samples, low_percentile=0.5, high_percentile=99.5):
    """Percentile-based min/max from sampled voxels."""
    info(f"Calculating percentile-based limits using {low_percentile}%-{high_percentile}% percentiles...")
    pmin, pmax = np.percentile(samples, [low_percentile, high_percentile])
    info(f"  - {low_percentile}% percentile (min): {pmin}")
    info(f"  - {high_percentile}% percentile (max): {pmax}")
    info(f"  - Absolute min: {samples.min()}")
    info(f"  - Absolute max: {samples.max()}")
    return pmin, pmax


def calculate_histogram_based_limits(samples, cumulative_threshold=0.02, bins=1000):
    """Limits from the cumulative histogram of sampled voxels."""
    info(f"Calculating histogram-based limits with {cumulative_threshold*100}% threshold...")
    min_val, max_val = float(samples.min()), float(samples.max())
    hist, edges = np.histogram(samples, bins=bins, range=(min_val, max_val))
    cumulative_norm = np.cumsum(hist) / hist.sum()
    centers = (edges[:-1] + edges[1:]) / 2
    hmin = centers[np.argmax(cumulative_norm >= cumulative_threshold)]
    hmax = centers[np.argmax(cumulative_norm >= (1 - cumulative_threshold))]
    info(f"Histogram-based limits: {hmin} to {hmax}")
    return hmin, hmax


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.argument('input_file', type=click.Path(exists=True, dir_okay=False))
@click.argument('output_path', type=click.Path())
@click.option('--dataset', type=str, default=None, help='Path of the 3D dataset inside the HDF5 file (e.g. /exchange/data). Default: /exchange/data if present, else the only 3D dataset.')
@click.option('--start', type=click.IntRange(min=0), default=0, help='First Z slice to convert (inclusive).')
@click.option('--end', type=click.IntRange(min=0), default=None, help='Last Z slice to convert (exclusive). Default: end of dataset.')
@click.option('--dtype', type=click.Choice(['int8', 'int16', 'int32', 'uint8', 'uint16', 'float32', 'float64']), default='uint8', help='Data type of the output.')
@click.option('--chunks', type=(int, int, int), default=(64, 64, 64), help='Chunk size for the Zarr array as a tuple of three integers.')
@click.option('--compression', type=click.Choice(['blosclz', 'lz4', 'lz4hc', 'zlib', 'zstd']), default='blosclz', help='Compression algorithm to use for Zarr storage.')
@click.option('--pixel_size', type=float, default=1.0, help='Pixel size in micrometers.')
@click.option('--chunk_size', type=int, default=64, help='Number of Z slices to read from HDF5 per processing chunk.')
@click.option('--verbose', is_flag=True, help='Enable verbose logging.')
@click.option('--min_percentile', type=float, default=0.5, help='Lower percentile for percentile-based min/max calculation (default: 0.5).')
@click.option('--max_percentile', type=float, default=99.5, help='Upper percentile for percentile-based min/max calculation (default: 99.5).')
@click.option('--use_histogram', is_flag=True, help='Use histogram-based method instead of percentile method.')
@click.option('--sample_slices', type=click.IntRange(min=0), default=256, show_default=True, help='Number of evenly spaced Z slices read to compute automatic min/max. 0 = all slices.')
@click.option('--max_samples', type=click.IntRange(min=1000), default=20_000_000, show_default=True, help='Max voxels kept for automatic min/max (pixel stride in Y/X is chosen to fit).')
@click.option('--min_val', type=float, default=None, help='Manually specify the minimum value for scaling. If provided, automatic calculation will be skipped.')
@click.option('--max_val', type=float, default=None, help='Manually specify the maximum value for scaling. If provided, automatic calculation will be skipped.')
@click.option('--levels', type=click.IntRange(min=0), default=0, help='Number of pyramid levels. 0 (default) = pick automatically from chunk_size + image dims.')
@click.option('--zarr-format', 'zarr_format', type=click.Choice(['2', '3']), default='3', show_default=True, help='Zarr on-disk format to write. v3 requires zarr>=3.')
@click.option('--shard', 'shard', type=(int, int, int), default=None, help='Zarr v3 shard shape (Z Y X) in voxels. Each dim must be a multiple of --chunks. Requires --zarr-format 3.')
@click.option('--normalize-slices', 'normalize_slices', type=str, default='none', show_default=True,
              help="Per-slice intensity normalization applied BEFORE scale_to_dtype: "
                   "'none', 'median', 'mean', 'mode', or 'pNN' percentile (e.g. p95).")
@click.option('--norm-reference', 'norm_reference', type=click.Choice(['median', 'mean']),
              default='median', show_default=True,
              help='How to combine per-slice statistics into the normalization reference.')
def main(input_file, output_path, dataset, start, end, dtype, chunks, compression, pixel_size,
         chunk_size, verbose, min_percentile, max_percentile, use_histogram, sample_slices, max_samples,
         min_val, max_val, levels, zarr_format, shard, normalize_slices, norm_reference):
    """
    Convert a 3D dataset from an HDF5 file into a multiscale OME-Zarr store.
    """
    setup_custom_logger(verbose=verbose)
    zarr_format = int(zarr_format)
    if zarr_format == 3 and ZARR_MAJOR < 3:
        raise click.BadParameter(
            f"--zarr-format 3 requires zarr>=3; installed is {zarr.__version__}"
        )
    if shard is not None:
        if zarr_format != 3:
            raise click.BadParameter("--shard requires --zarr-format 3; v2 has no sharding")
        for i, (s, c) in enumerate(zip(shard, chunks)):
            if s % c != 0:
                raise click.BadParameter(
                    f"--shard dim {i} ({s}) must be a multiple of --chunks dim {i} ({c})"
                )
    info(f"=== Writing Zarr v{zarr_format} (library zarr=={zarr.__version__}) ===")
    if shard is not None:
        info(f"Sharding ON: shard shape = {shard} voxels "
             f"({tuple(s // c for s, c in zip(shard, chunks))} chunks per shard)")

    dtype_map = {
        'int8': np.int8, 'int16': np.int16, 'int32': np.int32,
        'uint8': np.uint8, 'uint16': np.uint16,
        'float32': np.float32, 'float64': np.float64
    }
    target_dtype = dtype_map[dtype]

    with h5py.File(input_file, 'r') as h5:
        dset = find_dataset(h5, dataset)
        if dset.ndim != 3:
            raise click.ClickException(f"Dataset {dset.name} has shape {dset.shape}; expected 3D (Z, Y, X)")
        nz, real_y, real_x = dset.shape
        z0 = start
        z1 = nz if end is None else min(end, nz)
        if z0 >= z1:
            raise click.BadParameter(f"Empty slice range [{z0}, {z1}) for dataset with {nz} slices")
        info(f"Input: {input_file}:{dset.name} shape={dset.shape} dtype={dset.dtype} "
             f"hdf5-chunks={dset.chunks}; converting Z [{z0}, {z1})")

        if normalize_slices.lower() == 'none':
            scales = None
        else:
            try:
                scales = compute_slice_scales(dset, z0, z1, chunk_size, normalize_slices, norm_reference)
            except ValueError as e:
                raise click.BadParameter(str(e))

        if min_val is not None and max_val is not None:
            if min_val >= max_val:
                raise click.BadParameter(f"min_val ({min_val}) must be less than max_val ({max_val})")
            data_min, data_max = min_val, max_val
            info(f"Using manually specified data range: [{data_min}, {data_max}]")
        elif min_val is not None or max_val is not None:
            raise click.BadParameter("Both --min_val and --max_val must be provided together, or neither")
        else:
            samples = sample_volume(dset, z0, z1, sample_slices, max_samples, scales=scales)
            if use_histogram:
                data_min, data_max = calculate_histogram_based_limits(samples)
            else:
                data_min, data_max = calculate_percentile_limits(samples, min_percentile, max_percentile)
            del samples

        info(f"Using data range: [{data_min}, {data_max}] for {dtype} conversion")
        info(f"Using chunks: {chunks}")

        if levels > 0:
            target_levels = levels
            multiple = 2 ** target_levels
            if chunk_size % multiple != 0:
                new_cs = ((chunk_size + multiple - 1) // multiple) * multiple
                info(f"Rounding --chunk_size {chunk_size} -> {new_cs} (multiple of {multiple} for --levels {target_levels})")
                chunk_size = new_cs
            info(f"Pyramid: {target_levels} levels (padding chunks for computation, cropping to real extent on write)")
        else:
            target_levels = calculate_levels((chunk_size, real_y, real_x))
            info(f"Pyramid: {target_levels} levels (auto from chunk_size={chunk_size}, Y={real_y}, X={real_x})")

        n_slices = z1 - z0
        estimated_chunks = (n_slices + chunk_size - 1) // chunk_size
        info(f"Processing {n_slices} slices in chunks of {chunk_size}")

        mode = 'w'
        total_chunks_processed = 0
        with tqdm(total=estimated_chunks, desc="Processing chunks", unit="chunks") as chunk_pbar:
            for off, stack in iter_blocks(dset, z0, z1, chunk_size):
                info(f"Processing chunk {total_chunks_processed + 1}, shape: {stack.shape}")

                if scales is not None:
                    stack *= scales[off:off + stack.shape[0], None, None].astype(np.float32)

                stack_scaled = scale_to_dtype(stack, data_min, data_max, target_dtype)
                real_shape = stack_scaled.shape

                padded = pad_for_pyramid(stack_scaled, target_levels)
                if padded.shape != real_shape:
                    info(f"  - Padded {real_shape} -> {padded.shape} (for pyramid compute only)")

                info(f"Chunk stats: min={np.min(stack_scaled)} max={np.max(stack_scaled)} dtype={stack_scaled.dtype}")

                save_zarr(
                    padded, output_path, chunks, compression, pixel_size, mode,
                    original_dtype=target_dtype,
                    target_levels=target_levels,
                    real_shape=real_shape,
                    zarr_format=zarr_format,
                    shards=shard,
                )
                mode = 'a'
                total_chunks_processed += 1
                chunk_pbar.update(1)

    info(f"Processing complete! Total chunks processed: {total_chunks_processed}")
    info(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
