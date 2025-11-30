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

def calculate_levels(data):
   get_divisions = lambda n: (n & -n).bit_length() - 1
   dim = []
   sh = data.shape
   for i in range(0, len(sh)):
       dim.append(get_divisions(sh[i]))
   return min(dim)

def calculate_percentile_limits(input_dir, low_percentile=0.5, high_percentile=99.5, sample_ratio=100):
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
       for filename in tiff_files:
           filepath = os.path.join(input_dir, filename)
           try:
               img = tifffile.imread(filepath)
               # Sample pixels to save memory - take every sample_ratio-th pixel
               sample = img.flatten()[::sample_ratio]
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

def save_zarr(volume, output_path, chunks, compression, pixel_size, mode='w', original_dtype=np.uint8, shard_size=None):
   """
   Save volume to Zarr v3 format with optional sharding.

   Parameters:
   - volume: numpy array to save
   - output_path: path to output zarr store
   - chunks: tuple of chunk sizes
   - compression: compression algorithm name
   - pixel_size: pixel size in micrometers
   - mode: 'w' for write (new), 'a' for append
   - original_dtype: target data type
   - shard_size: tuple of shard sizes (None to disable sharding)
   """
   compressor = Blosc(cname=compression, clevel=1, shuffle=Blosc.BITSHUFFLE)

   if mode == 'w':
       if os.path.exists(output_path):
           shutil.rmtree(output_path)
       root_group = zarr.open_group(output_path, mode='w', zarr_version=3)
       
       # Calculate pyramid levels and create datasets metadata
       levels = calculate_levels(volume)
       if levels > 6:
           levels = 6
       
       pyramid_levels, _ = downsample(volume, levels)
       datasets = []
       
       for level, data in enumerate(pyramid_levels):
           data = data.astype(original_dtype)
           dataset_name = f"{level}"

           # Determine chunk/shard configuration for this level
           if shard_size is not None:
               # With sharding: use shard_size as chunk
               level_chunks = shard_size
           else:
               level_chunks = chunks

           z = root_group.create_dataset(
               name=dataset_name, shape=data.shape, chunks=level_chunks, dtype=data.dtype, compressor=compressor
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
       
       # Set metadata only once when creating new zarr file
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
       
   else:
       root_group = zarr.open_group(output_path, mode='a', zarr_version=3)
       
       levels = calculate_levels(volume)
       if levels > 6:
           levels = 6
       
       pyramid_levels, _ = downsample(volume, levels)
       
       for level, data in enumerate(pyramid_levels):
           data = data.astype(original_dtype)
           dataset_name = f"{level}"
           
           if dataset_name in root_group:
               z = root_group[dataset_name]
               # Resize array to accommodate new data
               current_shape = z.shape
               new_shape = (current_shape[0] + data.shape[0],) + current_shape[1:]
               z.resize(new_shape)
               # Write new data
               z[current_shape[0]:] = data

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
@click.option('--shard_size', type=(int, int, int), default=None, help='Shard size for Zarr v3 sharding as a tuple of three integers. If not specified, sharding is disabled.')
def main(input_dir, output_path, dtype, chunks, compression, pixel_size, chunk_size, verbose,
        min_percentile, max_percentile, use_histogram, sample_ratio, min_val, max_val, shard_size):
   """
   Main function to process TIFF images and save them as a Zarr v3 store with multiscale representations.

   Parameters:
   - input_dir (str): Path to the input directory containing TIFF images.
   - output_path (str): Path to the output Zarr store.
   - dtype (str): Data type of the images. Choices are 'int8', 'int16', 'int32', 'uint8', 'uint16', 'float32', 'float64'.
   - chunks (tuple of ints): Chunk size for the Zarr array as a tuple of three integers.
   - compression (str): Compression algorithm to use for Zarr storage. Choices are 'blosclz', 'lz4', 'lz4hc', 'zlib', 'zstd'.
   - pixel_size (float): Pixel size in micrometers.
   - chunk_size (int): Number of TIFF images to load in each chunk.
   - verbose (bool): Enable verbose logging.
   - min_percentile (float): Lower percentile for percentile-based min/max calculation.
   - max_percentile (float): Upper percentile for percentile-based min/max calculation.
   - use_histogram (bool): Use histogram-based method instead of percentile method.
   - sample_ratio (int): Sample every Nth pixel for percentile calculation.
   - min_val (float): Manually specify the minimum value for scaling.
   - max_val (float): Manually specify the maximum value for scaling.
   - shard_size (tuple of ints): Shard size for Zarr v3 sharding. If None, sharding is disabled.

   Returns:
   - None
   """
   setup_custom_logger(verbose=verbose)
   
   dtype_map = {
       'int8': np.int8, 'int16': np.int16, 'int32': np.int32, 
       'uint8': np.uint8, 'uint16': np.uint16, 
       'float32': np.float32, 'float64': np.float64
   }
   
   target_dtype = dtype_map[dtype]
   
   # Check if manual min/max are provided
   if min_val is not None and max_val is not None:
       if min_val >= max_val:
           raise ValueError(f"min_val ({min_val}) must be less than max_val ({max_val})")
       
       data_min = min_val
       data_max = max_val
       info(f"Using manually specified data range: [{data_min}, {data_max}]")
   elif min_val is not None or max_val is not None:
       raise ValueError("Both --min_val and --max_val must be provided together, or neither should be provided")
   else:
       # Calculate percentile-based limits using chosen method
       if use_histogram:
           data_min, data_max = calculate_histogram_based_limits(input_dir)
       else:
           data_min, data_max = calculate_percentile_limits(
               input_dir, min_percentile, max_percentile, sample_ratio
           )
   
   info(f"Using data range: [{data_min}, {data_max}] for {dtype} conversion")

   # Log sharding configuration
   if shard_size is not None:
       info(f"Zarr v3 sharding enabled with shard size: {shard_size}")
       info(f"Chunk size (within shards): {chunks}")
   else:
       info(f"Zarr v3 sharding disabled, using direct chunks: {chunks}")

   start_index = 0
   mode = 'w'
   total_chunks_processed = 0
   
   # Count total files for progress tracking
   total_tiff_files = len([f for f in os.listdir(input_dir) if f.lower().endswith(('.tif', '.tiff'))])
   estimated_chunks = (total_tiff_files + chunk_size - 1) // chunk_size  # Ceiling division
   
   info(f"Processing {total_tiff_files} TIFF files in chunks of {chunk_size}")
   
   # Progress bar for main processing
   with tqdm(total=estimated_chunks, desc="Processing chunks", unit="chunks") as chunk_pbar:
       while True:
           # Load data in original dtype (usually higher precision)
           # Note: You may need to modify load_tiff_chunked to not apply normalization
           stack, start_index = load_tiff_chunked(input_dir, np.float32, chunk_size, start_index)
           
           if stack is None:
               break
               
           info(f"Processing chunk {total_chunks_processed + 1}, shape: {stack.shape}")
           
           # Apply scaling to target dtype
           stack_scaled = scale_to_dtype(stack, data_min, data_max, target_dtype)
           
           # Log some statistics about the conversion
           info(f"Chunk stats after conversion:")
           info(f"  - Min: {np.min(stack_scaled)}, Max: {np.max(stack_scaled)}")
           info(f"  - Data type: {stack_scaled.dtype}")
           info(f"  - Shape: {stack_scaled.shape}")
           
           save_zarr(stack_scaled, output_path, chunks, compression, pixel_size, mode, original_dtype=target_dtype, shard_size=shard_size)
           mode = 'a'
           total_chunks_processed += 1
           chunk_pbar.update(1)
           
           if start_index >= total_tiff_files:
               break
   
   info(f"Processing complete! Total chunks processed: {total_chunks_processed}")
   info(f"Output saved to: {output_path}")

if __name__ == "__main__":
   main()
