import os
import shutil
import numpy as np
import zarr
import click
from numcodecs import Blosc
from .utils import load_tiff_chunked, downsample, minmaxHisto
from .log import info, setup_custom_logger

def calculate_levels(data, max_levels=10, min_size_threshold=64):
    """
    Calculate optimal number of pyramid levels with debugging info
    """
    shape = data.shape
    info(f"Data shape: {shape}")
    
    # Method 1: Based on powers of 2 (your original approach)
    get_divisions = lambda n: (n & -n).bit_length() - 1
    power_of_2_levels = []
    for i, dim_size in enumerate(shape):
        levels = get_divisions(dim_size)
        power_of_2_levels.append(levels)
        info(f"Dimension {i} (size {dim_size}): {levels} power-of-2 levels")
    
    theoretical_max = min(power_of_2_levels)
    info(f"Theoretical max from power-of-2: {theoretical_max}")
    
    # Method 2: Based on size reduction (more practical)
    size_based_levels = 0
    min_size = min(shape)
    temp_size = min_size
    
    info(f"Starting with minimum dimension size: {min_size}")
    while temp_size > min_size_threshold and size_based_levels < max_levels:
        temp_size //= 2
        size_based_levels += 1
        info(f"Level {size_based_levels}: min dimension would be {temp_size}")
    
    info(f"Size-based levels: {size_based_levels}")
    
    # Use the more practical approach - always create at least 3-4 levels for good performance
    final_levels = max(3, min(theoretical_max, size_based_levels, max_levels))
    info(f"Final levels chosen: {final_levels}")
    
    return final_levels

def optimize_chunks_for_neuroglancer(shape, base_chunk_size=(64, 128, 128)):
    """Optimize chunk sizes based on data dimensions and Neuroglancer access patterns"""
    z, y, x = shape
    cz, cy, cx = base_chunk_size
    
    # Adjust chunk size to avoid very small edge chunks
    optimal_cz = min(cz, max(32, z // max(1, z // cz)))
    optimal_cy = min(cy, max(64, y // max(1, y // cy)))
    optimal_cx = min(cx, max(64, x // max(1, x // cx)))
    
    return (optimal_cz, optimal_cy, optimal_cx)

def save_zarr_optimized(volume, output_path, chunks, compression, compression_level, pixel_size, 
                       max_levels, mode='w', original_dtype=np.uint8):
    """
    Optimized zarr saving with better performance for Neuroglancer
    """
    store = zarr.DirectoryStore(output_path)
    
    # Use lighter compression for better decompression speed
    compressor = Blosc(cname=compression, clevel=compression_level, shuffle=2)

    if mode == 'w':
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        root_group = zarr.group(store=store)
    else:
        root_group = zarr.open(store=store, mode='a')

    # Calculate optimal levels and chunks
    levels = calculate_levels(volume, max_levels)
    optimized_chunks = optimize_chunks_for_neuroglancer(volume.shape, chunks)
    
    # Debug output
    info(f"Volume shape: {volume.shape}")
    info(f"Calculated levels: {levels}")
    
    info(f"Creating {levels} pyramid levels with optimized chunks: {optimized_chunks}")

    pyramid_levels, _ = downsample(volume, levels)
    datasets = []

    # Create all levels efficiently
    for level, data in enumerate(pyramid_levels):
        data = data.astype(original_dtype)
        dataset_name = f"{level}"
        
        # Calculate chunks for this level
        level_chunks = tuple(min(c, s) for c, s in zip(optimized_chunks, data.shape))
        
        if dataset_name in root_group:
            # For append mode, check if we need to resize
            z = root_group[dataset_name]
            current_shape = z.shape
            new_shape = (current_shape[0] + data.shape[0], *current_shape[1:])
            z.resize(new_shape)
            z[current_shape[0]:] = data
        else:
            # Create new dataset with optimized settings
            z = root_group.create_dataset(
                name=dataset_name, 
                shape=data.shape, 
                chunks=level_chunks, 
                dtype=data.dtype, 
                compressor=compressor,
                # Add these for better performance
                order='C',  # Row-major order is generally faster
                synchronizer=None  # Disable synchronization for single-threaded access
            )
            z[:] = data

        scale_factor = 2 ** level
        
        # Keep original translation logic for proper pixel alignment between levels
        datasets.append({
            "path": dataset_name,
            "coordinateTransformations": [
                {"type": "scale", "scale": [scale_factor, scale_factor, scale_factor]},
                {"type": "translation", "translation": [2**(level-1) - 0.5, 2**(level-1) - 0.5, 2**(level-1) - 0.5]}
            ]
        })

    # Enhanced OME-Zarr metadata for better Neuroglancer compatibility
    root_group.attrs["multiscales"] = [{
        "version": "0.4",
        "name": os.path.basename(output_path),
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
    
    # Add additional metadata for Neuroglancer
    root_group.attrs["omero"] = {
        "channels": [{
            "color": "ffffff",
            "window": {"start": 0, "end": int(np.iinfo(original_dtype).max)},
            "label": "data",
            "active": True
        }]
    }

def load_and_process_efficiently(input_dir, dtype_np, chunk_size, global_min, global_max):
    """
    More efficient loading that minimizes memory fragmentation
    """
    # Try to load entire volume at once if possible
    tiff_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.tif', '.tiff'))])
    
    if len(tiff_files) <= chunk_size * 2:  # If small enough, load all at once
        info("Loading entire volume at once for optimal chunking")
        full_stack, _ = load_tiff_chunked(input_dir, dtype_np, len(tiff_files), 0, global_min, global_max)
        return [full_stack] if full_stack is not None else []
    else:
        # Load in chunks but try to align with final chunk boundaries
        info(f"Loading in chunks of {chunk_size} for large dataset")
        chunks = []
        start_index = 0
        
        while start_index < len(tiff_files):
            stack, start_index = load_tiff_chunked(input_dir, dtype_np, chunk_size, start_index, global_min, global_max)
            if stack is None:
                break
            chunks.append(stack)
            
        return chunks

@click.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--dtype', type=click.Choice(['int8', 'int16', 'int32', 'uint8', 'uint16', 'float32', 'float64']), 
              default='uint8', help='Data type of the images.')
@click.option('--chunks', type=(int, int, int), default=(64, 128, 128), 
              help='Chunk size for the Zarr array as (z, y, x). Optimized default for Neuroglancer.')
@click.option('--compression', type=click.Choice(['blosclz', 'lz4', 'lz4hc', 'zlib', 'zstd']), 
              default='lz4', help='Compression algorithm. lz4 recommended for speed.')
@click.option('--compression_level', type=int, default=1, 
              help='Compression level (1-9). Lower is faster decompression.')
@click.option('--force_levels', type=int, default=None, 
              help='Force a specific number of pyramid levels (overrides automatic calculation).')
@click.option('--max_levels', type=int, default=10, 
              help='Maximum number of pyramid levels to create.')
@click.option('--force_levels', type=int, default=None, 
              help='Force a specific number of pyramid levels (overrides automatic calculation).')
@click.option('--pixel_size', type=float, default=1.0, help='Pixel size in micrometers.')
@click.option('--chunk_size', type=int, default=64, help='Number of TIFF images to load in each chunk.')
@click.option('--verbose', is_flag=True, help='Enable verbose logging.')
@click.option('--optimize_loading', is_flag=True, default=True, 
              help='Optimize loading strategy for better chunk alignment.')
def main(input_dir, output_path, dtype, chunks, compression, compression_level, max_levels, 
         force_levels, pixel_size, chunk_size, verbose, optimize_loading):
    """
    Optimized TIFF to Zarr converter for fast Neuroglancer visualization.
    
    Key optimizations:
    - Better chunk sizes for Neuroglancer access patterns
    - Lighter compression for faster decompression
    - More pyramid levels for smooth zooming
    - Improved OME-Zarr metadata
    - Optimized loading strategies
    """
    setup_custom_logger(verbose=verbose)
    
    dtype_map = {
        'int8': np.int8, 'int16': np.int16, 'int32': np.int32, 
        'uint8': np.uint8, 'uint16': np.uint16, 
        'float32': np.float32, 'float64': np.float64
    }
    
    if force_levels is not None:
        info(f"Forcing {force_levels} pyramid levels")
        max_levels = force_levels
    
    info(f"Starting optimized conversion with settings:")
    info(f"  Chunks: {chunks}")
    info(f"  Compression: {compression} (level {compression_level})")
    info(f"  Max pyramid levels: {max_levels}")
    
    global_min, global_max = minmaxHisto(input_dir)
    info(f"Global min and max found: {global_min}, {global_max}")
    
    dtype_np = dtype_map[dtype]
    
    if optimize_loading:
        # Use optimized loading strategy
        volume_chunks = load_and_process_efficiently(input_dir, dtype_np, chunk_size, global_min, global_max)
        
        mode = 'w'
        for i, stack in enumerate(volume_chunks):
            info(f"Processing chunk {i+1}/{len(volume_chunks)}")
            save_zarr_optimized(stack, output_path, chunks, compression, compression_level, 
                              pixel_size, max_levels, mode, dtype_np)
            mode = 'a'
    else:
        # Use original chunked approach
        start_index = 0
        mode = 'w'
        
        while True:
            stack, start_index = load_tiff_chunked(input_dir, dtype_np, chunk_size, start_index, global_min, global_max)
            if stack is None:
                break
            save_zarr_optimized(stack, output_path, chunks, compression, compression_level,
                              pixel_size, max_levels, mode, dtype_np)
            mode = 'a'
            if start_index >= len(os.listdir(input_dir)):
                break
    
    info(f"Conversion complete! Zarr store saved to: {output_path}")
    info("Optimizations applied:")
    info("  ✓ Neuroglancer-optimized chunk sizes")
    info("  ✓ Fast decompression settings")
    info("  ✓ Extended pyramid levels")
    info("  ✓ Enhanced OME-Zarr metadata")

if __name__ == "__main__":
    main()
