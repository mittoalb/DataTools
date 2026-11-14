import os
import shutil
import numpy as np
import zarr
import click
import tifffile
from numcodecs import Blosc
from tqdm import tqdm
from .utils import load_tiff_chunked, downsample
from .log import info, setup_custom_logger

ZARR_MAJOR = int(zarr.__version__.split('.')[0])


def pad_for_pyramid(arr, target_levels):
    """Edge-pad a 3D array so each dim is a multiple of 2**target_levels.

    Only used for pyramid COMPUTATION; callers must crop results back to the
    real extent before writing to storage so padded voxels never persist.
    """
    if target_levels <= 0:
        return arr
    multiple = 2 ** target_levels
    pads = tuple((0, (multiple - s % multiple) % multiple) for s in arr.shape)
    if all(p == (0, 0) for p in pads):
        return arr
    return np.pad(arr, pads, mode='edge')


def calculate_levels(shape):
    """Max N such that every dim is divisible by 2**N."""
    get_divisions = lambda n: (n & -n).bit_length() - 1 if n > 0 else 0
    return min(get_divisions(s) for s in shape)


def level_shape(real_shape, level):
    """Shape of pyramid level L given the real (unpadded) base shape."""
    f = 2 ** level
    return tuple(max(1, (s + f - 1) // f) for s in real_shape)

def _make_slice_stat_fn(spec, hist_bins=256):
    """Return a function img -> float for per-slice intensity statistic.

    spec: 'median', 'mean', 'mode' (histogram peak), or 'pNN' (percentile).
    """
    s = spec.lower()
    if s == 'mean':
        return lambda img: float(img.mean())
    if s == 'median':
        return lambda img: float(np.median(img))
    if s == 'mode':
        def _mode(img):
            flat = img.ravel()
            lo = float(flat.min()); hi = float(flat.max())
            if hi <= lo:
                return float(lo)
            counts, edges = np.histogram(flat, bins=hist_bins, range=(lo, hi))
            k = int(np.argmax(counts))
            return float(0.5 * (edges[k] + edges[k + 1]))
        return _mode
    if s.startswith('p'):
        try:
            q = float(s[1:])
        except ValueError:
            raise ValueError(f"--normalize-slices {spec!r} not understood")
        if not 0 < q < 100:
            raise ValueError("Percentile must be in (0, 100)")
        return lambda img: float(np.percentile(img, q))
    raise ValueError(f"--normalize-slices {spec!r} not understood; use median, mean, mode, or pNN")


def _combine_stats(values, how):
    arr = np.asarray(values, dtype=np.float64)
    if how == 'median':
        return float(np.median(arr))
    if how == 'mean':
        return float(np.mean(arr))
    raise ValueError(f"--norm-reference {how!r} not understood; use median or mean")


def compute_slice_scales(input_dir, stat_spec, reference_how):
    """One pass over the TIFF stack: return per-file multiplicative scale factors
    that flatten slice-to-slice intensity drift.

    Index order matches sorted() of the TIFF filenames (same ordering used by
    load_tiff_chunked). Files whose stat is <= 0 get scale=1.0 and a warning.
    """
    stat_fn = _make_slice_stat_fn(stat_spec)
    tiff_files = sorted(f for f in os.listdir(input_dir)
                        if f.lower().endswith(('.tif', '.tiff')))
    info(f"Normalization pre-pass: computing per-slice {stat_spec} for {len(tiff_files)} files...")
    stats = np.empty(len(tiff_files), dtype=np.float64)
    with tqdm(total=len(tiff_files), desc=f"Slice {stat_spec}", unit="files") as pbar:
        for i, fn in enumerate(tiff_files):
            stats[i] = stat_fn(tifffile.imread(os.path.join(input_dir, fn)))
            pbar.update(1)
    reference = _combine_stats(stats, reference_how)
    if reference <= 0:
        raise ValueError(f"Reference ({reference_how} of per-slice {stat_spec}) is {reference}; cannot normalize")
    nonpos = int((stats <= 0).sum())
    scales = np.where(stats > 0, reference / stats, 1.0)
    spread_pct = (stats.max() - stats.min()) / reference * 100 if reference > 0 else 0.0
    info(f"Per-slice {stat_spec}: min={stats.min():.4f} max={stats.max():.4f} "
         f"spread={spread_pct:.1f}% of {reference_how} reference ({reference:.4f})")
    info(f"Scale factors: min={scales.min():.4f} max={scales.max():.4f}")
    if nonpos:
        info(f"Warning: {nonpos} slices had {stat_spec} <= 0; left unscaled")
    return scales


def calculate_percentile_limits(input_dir, low_percentile=0.5, high_percentile=99.5, sample_ratio=100, scales=None):
   """
   Calculate percentile-based min/max to exclude outliers.
   
   Parameters:
   - input_dir: Directory containing TIFF files
   - low_percentile: Lower percentile threshold (default 0.5%)
   - high_percentile: Upper percentile threshold (default 99.5%)
   - sample_ratio: Sample every Nth pixel to save memory (default every 100th)
   
   Returns:
   - percentile_min, percentile_max: Percentile-based min and max values
   """
   info(f"Calculating percentile-based limits using {low_percentile}%-{high_percentile}% percentiles...")
   
   all_values = []
   file_count = 0
   
   # Get all TIFF files
   tiff_files = [f for f in sorted(os.listdir(input_dir)) 
                 if f.lower().endswith(('.tif', '.tiff'))]
   
   total_files = len(tiff_files)
   info(f"Processing {total_files} TIFF files for percentile calculation...")
   
   # Progress bar for percentile calculation
   with tqdm(total=total_files, desc="Calculating percentiles", unit="files") as pbar:
       for i, filename in enumerate(tiff_files):
           filepath = os.path.join(input_dir, filename)
           try:
               img = tifffile.imread(filepath)
               # Sample pixels to save memory - take every sample_ratio-th pixel
               sample = img.flatten()[::sample_ratio]
               if scales is not None:
                   # Reflect the per-slice intensity normalization in the sampled
                   # values so data_min/data_max match what scale_to_dtype sees.
                   sample = sample.astype(np.float32) * float(scales[i])
               all_values.extend(sample)
               file_count += 1

           except Exception as e:
               info(f"Warning: Could not read {filename}: {e}")

           pbar.update(1)
   
   if not all_values:
       raise ValueError("No valid TIFF files found or could be read")
   
   info("Computing percentiles from sampled data...")
   all_values = np.array(all_values)
   percentile_min = np.percentile(all_values, low_percentile)
   percentile_max = np.percentile(all_values, high_percentile)
   
   info(f"Percentile limits calculated from {file_count} files:")
   info(f"  - {low_percentile}% percentile (min): {percentile_min}")
   info(f"  - {high_percentile}% percentile (max): {percentile_max}")
   info(f"  - Absolute min: {np.min(all_values)}")
   info(f"  - Absolute max: {np.max(all_values)}")
   
   return percentile_min, percentile_max

def calculate_histogram_based_limits(input_dir, cumulative_threshold=0.02, bins=1000):
   """
   Calculate limits based on cumulative histogram analysis.
   
   Parameters:
   - input_dir: Directory containing TIFF files
   - cumulative_threshold: Threshold for cumulative histogram (default 2%)
   - bins: Number of histogram bins
   
   Returns:
   - histogram_min, histogram_max: Histogram-based min and max values
   """
   info(f"Calculating histogram-based limits with {cumulative_threshold*100}% threshold...")
   
   all_histograms = []
   min_val = float('inf')
   max_val = float('-inf')
   
   tiff_files = [f for f in sorted(os.listdir(input_dir)) 
                 if f.lower().endswith(('.tif', '.tiff'))]
   
   info("Finding data range from sample files...")
   # First pass: find global min/max for histogram range (sample first 10 files)
   sample_files = tiff_files[:min(10, len(tiff_files))]
   with tqdm(total=len(sample_files), desc="Sampling range", unit="files") as pbar:
       for filename in sample_files:
           filepath = os.path.join(input_dir, filename)
           try:
               img = tifffile.imread(filepath)
               min_val = min(min_val, np.min(img))
               max_val = max(max_val, np.max(img))
           except Exception as e:
               pass
           pbar.update(1)
   
   info(f"Building histograms for {len(tiff_files)} files...")
   # Second pass: build histograms
   with tqdm(total=len(tiff_files), desc="Building histograms", unit="files") as pbar:
       for filename in tiff_files:
           filepath = os.path.join(input_dir, filename)
           try:
               img = tifffile.imread(filepath)
               hist, _ = np.histogram(img.flatten(), bins=bins, range=(min_val, max_val))
               all_histograms.append(hist)
           except Exception as e:
               pass
           pbar.update(1)
   
   # Sum all histograms
   total_hist = np.sum(all_histograms, axis=0)
   cumulative = np.cumsum(total_hist)
   cumulative_norm = cumulative / cumulative[-1]
   
   # Create bin centers
   bin_edges = np.linspace(min_val, max_val, bins + 1)
   bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
   
   # Find values at cumulative thresholds
   min_idx = np.argmax(cumulative_norm >= cumulative_threshold)
   max_idx = np.argmax(cumulative_norm >= (1 - cumulative_threshold))
   
   histogram_min = bin_centers[min_idx]
   histogram_max = bin_centers[max_idx]
   
   info(f"Histogram-based limits: {histogram_min} to {histogram_max}")
   
   return histogram_min, histogram_max

def scale_to_dtype(data, min_val, max_val, target_dtype):
   """
   Scale and convert data to target dtype with proper contrast stretching.
   
   Parameters:
   - data: Input data array
   - min_val: Minimum value for scaling
   - max_val: Maximum value for scaling
   - target_dtype: Target numpy dtype
   
   Returns:
   - scaled: Data scaled and converted to target dtype
   """
   # Clip values to min/max range
   data_clipped = np.clip(data, min_val, max_val)
   
   # Avoid division by zero
   range_val = max_val - min_val
   if range_val == 0:
       info("Warning: min_val equals max_val, returning zeros")
       return np.zeros_like(data, dtype=target_dtype)
   
   # Scale based on target dtype
   if target_dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
       # Unsigned integer types
       type_info = np.iinfo(target_dtype)
       scaled = ((data_clipped - min_val) / range_val * type_info.max).astype(target_dtype)
   elif target_dtype in [np.int8, np.int16, np.int32, np.int64]:
       # Signed integer types
       type_info = np.iinfo(target_dtype)
       scaled = ((data_clipped - min_val) / range_val * 
                (type_info.max - type_info.min) + type_info.min).astype(target_dtype)
   else:
       # Float types - normalize to 0-1 range
       scaled = ((data_clipped - min_val) / range_val).astype(target_dtype)
   
   return scaled

def _open_group(output_path, mode, zarr_format):
    """Open a Zarr group of the requested format.

    zarr_format=2: works on both zarr 2.x and 3.x libraries.
    zarr_format=3: requires zarr >= 3.
    """
    if zarr_format == 3 and ZARR_MAJOR < 3:
        raise RuntimeError(
            f"--zarr-format 3 requires zarr>=3; installed is {zarr.__version__}"
        )
    if ZARR_MAJOR >= 3:
        return zarr.open_group(output_path, mode=mode, zarr_format=zarr_format)
    # zarr 2.x: only v2 output is possible
    return zarr.open_group(output_path, mode=mode)


def _make_compressor(compression, zarr_format):
    """Return the appropriate blosc compressor for the chosen output format.

    v2 uses numcodecs.Blosc (works on both zarr 2 and 3 libraries writing v2
    stores). v3 requires a zarr-native BytesBytesCodec from zarr.codecs.

    shuffle='shuffle' (byte-shuffle) matches the pre-modification writer and
    is what Neuroglancer's Blosc decoder currently handles reliably on v3
    stores. bitshuffle was previously here but caused Neuroglancer to fail
    to decode chunks (surfaced in the UI as "no zarr metadata found").
    """
    if zarr_format == 3:
        from zarr.codecs import BloscCodec  # local import: only exists on zarr>=3
        return BloscCodec(cname=compression, clevel=1, shuffle='shuffle')
    # numcodecs.Blosc shuffle codes: 0=NOSHUFFLE, 1=SHUFFLE (byte), 2=BITSHUFFLE
    return Blosc(cname=compression, clevel=1, shuffle=1)


def _create_array(group, name, shape, chunks, dtype, compressor, shards=None):
    """Create an array under group in whichever format the group was opened as.

    shards (zarr v3 only): shard shape in voxels; must be a multiple of chunks.
    Ignored (with no error) when writing v2 output or when shards is None.
    """
    if ZARR_MAJOR >= 3:
        kwargs = dict(
            name=name, shape=shape, chunks=chunks, dtype=dtype,
            compressors=[compressor],
        )
        if shards is not None:
            kwargs["shards"] = shards
        return group.create_array(**kwargs)
    return group.create_dataset(
        name=name, shape=shape, chunks=chunks, dtype=dtype, compressor=compressor
    )


def _resolve_level_shards(user_shards, level_chunks, level_shape):
    """Compute a usable shard shape for one pyramid level, or None.

    - Axis 0 (Z) is NOT clamped to level_shape[0]: 'w' mode creates the array
      with only the first chunk's Z size, then 'a' mode grows it via resize.
      The shard Z dim must survive that growth, so we trust the user's choice.
    - Axes 1,2 (Y/X) ARE clamped to level_shape (they're fixed per level).
    - Each dim snapped down to a multiple of its chunk (zarr v3 requirement).
    - If every dim collapses to its chunk size, sharding is pointless: None.
    """
    if user_shards is None:
        return None
    out = []
    for axis, (s, c, ext) in enumerate(zip(user_shards, level_chunks, level_shape)):
        if axis > 0:
            s = min(s, ext)
        s = (s // c) * c
        if s <= 0:
            return None
        out.append(s)
    if all(s == c for s, c in zip(out, level_chunks)):
        return None
    return tuple(out)


def save_zarr(volume, output_path, chunks, compression, pixel_size, mode='w',
              original_dtype=np.uint8, target_levels=None, real_shape=None,
              zarr_format=2, shards=None):
    """Append (or create) a multiscale pyramid in a Zarr store.

    Parameters
    ----------
    volume : ndarray
        The chunk to write. May be edge-padded along Z/Y/X so pyramid
        computation is clean; everything past ``real_shape`` is discarded
        before anything hits storage.
    target_levels : int
        Number of pyramid levels to produce. Fixed for the WHOLE run (do not
        recompute per chunk); on 'w' it is stored in root attrs, on 'a' it is
        read back from there and must match what the caller passes.
    real_shape : (Z, Y, X)
        The chunk's true (unpadded) extent. Each pyramid level L is cropped
        to ceil(real_shape / 2**L) before writing.
    zarr_format : {2, 3}
        Which on-disk format to write. v2 works on zarr>=2; v3 requires zarr>=3.
    shards : (Z, Y, X) or None
        Shard shape in voxels for Zarr v3 (packs many chunks into one file).
        Must be a multiple of chunks; ignored on v2. Automatically clamped
        and chunk-snapped per pyramid level by _resolve_level_shards.
    """
    if target_levels is None or real_shape is None:
        raise ValueError("save_zarr now requires target_levels and real_shape")
    if shards is not None and zarr_format != 3:
        raise ValueError("shards require --zarr-format 3; v2 has no sharding")

    compressor = _make_compressor(compression, zarr_format)

    if mode == 'w':
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        root_group = _open_group(output_path, mode='w', zarr_format=zarr_format)
        # NOTE: we deliberately do NOT stamp zarr_format / target_levels /
        # shards into attrs — the pre-modification writer only wrote
        # "multiscales" at the root, and strict OME-Zarr readers (notably
        # Neuroglancer) reject the file when extra keys show up next to it.
    else:
        root_group = _open_group(output_path, mode='a', zarr_format=zarr_format)
        # Append-mode consistency check: derive target_levels from the existing
        # array keys (one per pyramid level, 0..N) instead of a stored attr.
        existing = sorted(k for k in root_group.array_keys())
        if existing:
            stored = len(existing) - 1
            if stored != int(target_levels):
                raise ValueError(
                    f"target_levels mismatch: store has {stored} (arrays {existing}), "
                    f"chunk expects {target_levels}"
                )

    pyramid_levels, _ = downsample(volume, target_levels)
    datasets_meta = []

    for level, data in enumerate(pyramid_levels):
        dataset_name = f"{level}"
        level_real = level_shape(real_shape, level)
        # Crop padded pyramid output back to the chunk's real extent so
        # edge-padding never reaches storage.
        slicer = tuple(slice(0, s) for s in level_real)
        data = data[slicer].astype(original_dtype)

        if mode == 'w':
            # Pre-allocate with just this (first) chunk's real level shape;
            # subsequent 'a' calls will resize Z as more chunks arrive.
            level_chunks = tuple(min(c, s) for c, s in zip(chunks, data.shape))
            level_shards = _resolve_level_shards(shards, level_chunks, data.shape)
            z = _create_array(
                root_group, dataset_name, data.shape, level_chunks, data.dtype,
                compressor, shards=level_shards,
            )
            z[:] = data

            scale_factor = 2 ** level
            # Correct downscale_local_mean grid offset: level-L voxel centers
            # sit at ((2**L - 1)/2) in level-0 units along each axis.
            offset = (2 ** level - 1) / 2.0
            datasets_meta.append({
                "path": dataset_name,
                "coordinateTransformations": [
                    {"type": "scale",
                     "scale": [pixel_size * scale_factor] * 3},
                    {"type": "translation",
                     "translation": [pixel_size * offset] * 3},
                ],
            })
        else:
            if dataset_name not in root_group:
                # Fixed-depth pyramid means every chunk emits every level;
                # a missing one would be a programming error, not a silent skip.
                raise RuntimeError(
                    f"Pyramid level {dataset_name} missing from existing store"
                )
            z = root_group[dataset_name]
            cur = z.shape
            z.resize((cur[0] + data.shape[0],) + cur[1:])
            z[cur[0]:] = data

    if mode == 'w':
        # OME-NGFF 0.4 metadata at the root attributes works for BOTH Zarr v2
        # AND v3 in Neuroglancer and other common readers (napari, viv).
        # Officially NGFF 0.5 is the spec-pinned pair for Zarr v3 (nested under
        # an "ome" key), but Neuroglancer's reader currently prefers / expects
        # the 0.4-at-root layout even on v3 stores, so we keep that for now.
        multiscales = [{
            "version": "0.4",
            "name": os.path.basename(output_path.rstrip("/")) or "image",
            "axes": [
                {"name": "z", "type": "space", "unit": "micrometer"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"},
            ],
            "datasets": datasets_meta,
            "type": "mean",
            "metadata": {
                "method": "skimage.transform.downscale_local_mean",
                "kwargs": {"factors": [2, 2, 2]},
            },
        }]
        root_group.attrs["multiscales"] = multiscales

@click.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--dtype', type=click.Choice(['int8', 'int16', 'int32', 'uint8', 'uint16', 'float32', 'float64']), default='uint8', help='Data type of the images.')
@click.option('--chunks', type=(int, int, int), default=(64, 64, 64), help='Chunk size for the Zarr array as a tuple of three integers.')
@click.option('--compression', type=click.Choice(['blosclz', 'lz4', 'lz4hc', 'zlib', 'zstd']), default='blosclz', help='Compression algorithm to use for Zarr storage.')
@click.option('--pixel_size', type=float, default=1.0, help='Pixel size in micrometers.')
@click.option('--chunk_size', type=int, default=64, help='Number of TIFF images to load in each chunk.')
@click.option('--verbose', is_flag=True, help='Enable verbose logging.')
@click.option('--min_percentile', type=float, default=0.5, help='Lower percentile for percentile-based min/max calculation (default: 0.5).')
@click.option('--max_percentile', type=float, default=99.5, help='Upper percentile for percentile-based min/max calculation (default: 99.5).')
@click.option('--use_histogram', is_flag=True, help='Use histogram-based method instead of percentile method.')
@click.option('--sample_ratio', type=int, default=100, help='Sample every Nth pixel for percentile calculation (default: 100).')
@click.option('--min_val', type=float, default=None, help='Manually specify the minimum value for scaling. If provided, automatic calculation will be skipped.')
@click.option('--max_val', type=float, default=None, help='Manually specify the maximum value for scaling. If provided, automatic calculation will be skipped.')
@click.option('--levels', type=click.IntRange(min=0), default=0, help='Number of pyramid levels. 0 (default) = pick automatically from the first chunk + image dims.')
@click.option('--zarr-format', 'zarr_format', type=click.Choice(['2', '3']), default='2', show_default=True, help='Zarr on-disk format to write. v3 requires zarr>=3.')
@click.option('--shard', 'shard', type=(int, int, int), default=None, help='Zarr v3 shard shape (Z Y X) in voxels. Packs many chunks into one file; each dim must be a multiple of --chunks. Requires --zarr-format 3.')
@click.option('--normalize-slices', 'normalize_slices', type=str, default='none', show_default=True,
              help="Per-slice intensity normalization applied BEFORE scale_to_dtype. "
                   "'none' (default) disables. Otherwise: 'median', 'mean', 'mode' "
                   "(histogram peak -- robust when the object fills the frame), or "
                   "'pNN' percentile (e.g. p95). Each slice is scaled by "
                   "reference / stat(slice) where reference is combined across all slices.")
@click.option('--norm-reference', 'norm_reference', type=click.Choice(['median', 'mean']),
              default='median', show_default=True,
              help='How to combine per-slice statistics into the normalization reference.')
def main(input_dir, output_path, dtype, chunks, compression, pixel_size, chunk_size, verbose,
        min_percentile, max_percentile, use_histogram, sample_ratio, min_val, max_val, levels,
        zarr_format, shard, normalize_slices, norm_reference):
    """
    Convert a folder of TIFF images into a multiscale OME-Zarr store.
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

    if normalize_slices.lower() == 'none':
        scales = None
    else:
        if use_histogram:
            raise click.BadParameter(
                "--normalize-slices is not supported with --use_histogram; "
                "use the percentile-based limits path instead."
            )
        try:
            scales = compute_slice_scales(input_dir, normalize_slices, norm_reference)
        except ValueError as e:
            raise click.BadParameter(str(e))

    if min_val is not None and max_val is not None:
        if min_val >= max_val:
            raise ValueError(f"min_val ({min_val}) must be less than max_val ({max_val})")
        data_min, data_max = min_val, max_val
        info(f"Using manually specified data range: [{data_min}, {data_max}]")
    elif min_val is not None or max_val is not None:
        raise ValueError("Both --min_val and --max_val must be provided together, or neither should be provided")
    else:
        if use_histogram:
            data_min, data_max = calculate_histogram_based_limits(input_dir)
        else:
            data_min, data_max = calculate_percentile_limits(
                input_dir, min_percentile, max_percentile, sample_ratio, scales=scales
            )

    info(f"Using data range: [{data_min}, {data_max}] for {dtype} conversion")
    info(f"Using chunks: {chunks}")

    # Inspect input dims upfront so target_levels is fixed for the whole run
    # (prevents the per-chunk level drift from the old save_zarr).
    tiff_files = sorted(
        f for f in os.listdir(input_dir)
        if f.lower().endswith(('.tif', '.tiff'))
    )
    total_tiff_files = len(tiff_files)
    if total_tiff_files == 0:
        raise click.ClickException(f"No TIFF files found in {input_dir}")
    sample = tifffile.imread(os.path.join(input_dir, tiff_files[0]))
    real_y, real_x = sample.shape[-2:]

    if levels > 0:
        target_levels = levels
        multiple = 2 ** target_levels
        if chunk_size % multiple != 0:
            new_cs = ((chunk_size + multiple - 1) // multiple) * multiple
            info(f"Rounding --chunk_size {chunk_size} -> {new_cs} (multiple of {multiple} for --levels {target_levels})")
            chunk_size = new_cs
        info(f"Pyramid: {target_levels} levels (padding chunks for computation, cropping to real extent on write)")
    else:
        # Auto: pick the largest N such that every full chunk + image Y/X
        # is divisible by 2**N. Last-chunk Z is handled by the per-chunk
        # pad/crop in save_zarr regardless.
        target_levels = calculate_levels((chunk_size, real_y, real_x))
        info(f"Pyramid: {target_levels} levels (auto from chunk_size={chunk_size}, Y={real_y}, X={real_x})")

    estimated_chunks = (total_tiff_files + chunk_size - 1) // chunk_size
    info(f"Processing {total_tiff_files} TIFF files in chunks of {chunk_size}")

    start_index = 0
    mode = 'w'
    total_chunks_processed = 0

    with tqdm(total=estimated_chunks, desc="Processing chunks", unit="chunks") as chunk_pbar:
        while True:
            chunk_start = start_index
            stack, start_index = load_tiff_chunked(input_dir, np.float32, chunk_size, start_index)
            if stack is None:
                break

            info(f"Processing chunk {total_chunks_processed + 1}, shape: {stack.shape}")

            if scales is not None:
                chunk_scales = scales[chunk_start:chunk_start + stack.shape[0]].astype(np.float32)
                stack = stack * chunk_scales[:, None, None]

            stack_scaled = scale_to_dtype(stack, data_min, data_max, target_dtype)
            real_shape = stack_scaled.shape  # (real_cz, real_y, real_x) BEFORE padding

            # Pad for clean pyramid computation; save_zarr crops back to real_shape.
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

            if start_index >= total_tiff_files:
                break

    info(f"Processing complete! Total chunks processed: {total_chunks_processed}")
    info(f"Output saved to: {output_path}")
    info(f"=== Wrote Zarr v{zarr_format}. Inspect with: python -c 'import zarr; g=zarr.open(\"{output_path}\"); print(dict(g.attrs))' ===")


if __name__ == "__main__":
    main()
