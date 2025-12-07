#!/usr/bin/env python3
"""
Remove radial line artifacts from FFT using polar coordinate normalization.
Supports parallel processing of multiple files.

Usage:
    # Single file
    python remove_fft_artifacts_polar_parallel.py input.tiff --output filtered.tiff
    
    # Process folder with parallel threads
    python remove_fft_artifacts_polar_parallel.py input_folder/ --output output_folder/ --threads 4
"""

import numpy as np
from scipy import fftpack
import argparse
import tifffile
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import warnings


def cart2polar(y, x, center):
    """Convert cartesian to polar coordinates."""
    y_centered = y - center[0]
    x_centered = x - center[1]
    r = np.sqrt(x_centered**2 + y_centered**2)
    theta = np.arctan2(y_centered, x_centered)
    return r, theta


def polar2cart(r, theta, center):
    """Convert polar to cartesian coordinates."""
    x = r * np.cos(theta) + center[1]
    y = r * np.sin(theta) + center[0]
    return y, x


def remove_radial_artifacts_polar(image, dc_protect_radius=15, show_result=False, 
                                   output_path=None, radial_percentile=50, verbose=True):
    """
    Remove radial line artifacts by normalizing in polar coordinates.
    
    Parameters:
    -----------
    image : ndarray
        Input image (2D array)
    dc_protect_radius : int
        Radius around DC component to preserve (default: 15)
    show_result : bool
        Whether to display the results (default: False)
    output_path : str
        Path to save the filtered image (default: None)
    radial_percentile : float
        Percentile for radial normalization (default: 50, i.e., median)
    verbose : bool
        Print processing information (default: True)
    
    Returns:
    --------
    filtered_image : ndarray
        Filtered image with radial artifacts removed
    """
    
    # Store original dtype and range
    original_dtype = image.dtype
    original_min = float(image.min())
    original_max = float(image.max())
    
    if verbose:
        print(f"Input dtype: {original_dtype}, range: [{original_min:.2f}, {original_max:.2f}]")
    
    # Compute FFT
    fft = fftpack.fft2(image.astype(np.float64))  # Bug fix: ensure float for FFT
    fft_shift = fftpack.fftshift(fft)
    fft_mag = np.abs(fft_shift)
    fft_phase = np.angle(fft_shift)
    
    rows, cols = fft_mag.shape
    center = (rows // 2, cols // 2)
    
    # Create coordinate grids
    y, x = np.ogrid[:rows, :cols]
    r, theta = cart2polar(y, x, center)
    
    # Convert to polar coordinates
    max_radius = int(np.sqrt((rows/2)**2 + (cols/2)**2))
    n_theta = 360  # 1 degree resolution
    
    theta_bins = np.linspace(-np.pi, np.pi, n_theta + 1)
    r_bins = np.arange(0, max_radius + 1)
    
    # Digitize angles and radii
    theta_idx = np.digitize(theta, theta_bins) - 1
    theta_idx = np.clip(theta_idx, 0, n_theta - 1)  # Bug fix: ensure valid indices
    r_idx = r.astype(int)
    r_idx = np.clip(r_idx, 0, max_radius - 1)
    
    # Create polar representation using vectorized approach (much faster)
    # Bug fix: previous double loop was very slow
    polar_mag = np.zeros((max_radius, n_theta))
    polar_counts = np.zeros((max_radius, n_theta))
    
    # Flatten arrays for faster processing
    r_flat = r_idx.ravel()
    t_flat = theta_idx.ravel()
    mag_flat = fft_mag.ravel()
    
    # Use numpy's add.at for efficient accumulation
    np.add.at(polar_mag, (r_flat, t_flat), mag_flat)
    np.add.at(polar_counts, (r_flat, t_flat), 1)
    
    # Average
    with np.errstate(divide='ignore', invalid='ignore'):
        polar_mag = np.where(polar_counts > 0, polar_mag / polar_counts, 0)
    
    # Normalize each radial line by its percentile
    polar_normalized = np.zeros_like(polar_mag)
    for ri in range(dc_protect_radius, max_radius):
        radial_profile = polar_mag[ri, :]
        if radial_profile.max() > 0:
            # Bug fix: handle case when all values are zero
            valid_values = radial_profile[radial_profile > 0]
            if len(valid_values) > 0:
                radial_median = np.percentile(valid_values, radial_percentile)
                if radial_median > 0:
                    polar_normalized[ri, :] = radial_profile / radial_median
                else:
                    polar_normalized[ri, :] = radial_profile
            else:
                polar_normalized[ri, :] = radial_profile
        else:
            polar_normalized[ri, :] = radial_profile
    
    # Preserve DC and nearby region
    polar_normalized[:dc_protect_radius, :] = 1.0  # Bug fix: use 1.0 not raw polar_mag
    
    # Map back to cartesian - create normalization map (vectorized)
    # Bug fix: avoid another slow double loop
    normalization_map = np.ones((rows, cols))
    
    # Create mask for regions to normalize
    mask = r >= dc_protect_radius
    
    # Get normalization values for each pixel
    norm_values = polar_normalized[r_idx, theta_idx]
    
    # Apply mask
    normalization_map[mask] = norm_values[mask]
    
    # Bug fix: avoid division by zero
    normalization_map = np.where(normalization_map > 0, normalization_map, 1.0)
    
    # Apply normalization to FFT magnitude
    fft_mag_normalized = fft_mag / normalization_map
    
    # Reconstruct complex FFT
    fft_filtered = fft_mag_normalized * np.exp(1j * fft_phase)
    
    # Inverse FFT
    img_filtered = np.real(fftpack.ifft2(fftpack.ifftshift(fft_filtered)))
    
    # Bug fix: properly handle dtype conversion and range preservation
    if original_dtype == np.uint8:
        img_filtered = np.clip(img_filtered, 0, 255).astype(np.uint8)
    elif original_dtype == np.uint16:
        img_filtered = np.clip(img_filtered, 0, 65535).astype(np.uint16)
    else:
        img_filtered = img_filtered.astype(original_dtype)
    
    if verbose:
        print(f"Image shape: {rows}×{cols}")
        print(f"Protected DC radius: {dc_protect_radius}")
        print(f"Output range: [{float(img_filtered.min()):.2f}, {float(img_filtered.max()):.2f}]")
    
    if show_result:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(np.log(fft_mag + 1), cmap='hot')
        axes[0, 1].set_title('Original FFT (log scale)')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(polar_mag, aspect='auto', cmap='hot')
        axes[0, 2].set_title('Polar Representation')
        axes[0, 2].set_xlabel('Angle (bins)')
        axes[0, 2].set_ylabel('Radius (pixels)')
        
        axes[1, 0].imshow(normalization_map, cmap='RdBu_r', vmin=0.5, vmax=1.5)
        axes[1, 0].set_title('Normalization Map')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(np.log(fft_mag_normalized + 1), cmap='hot')
        axes[1, 1].set_title('Normalized FFT (log scale)')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(img_filtered, cmap='gray')
        axes[1, 2].set_title('Filtered Image')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    if output_path:
        tifffile.imwrite(output_path, img_filtered)
        if verbose:
            print(f"Saved to {output_path}")
    
    return img_filtered


def process_single_file(input_path, output_path, dc_protect, percentile, verbose=False):
    """
    Process a single file.
    
    Returns:
    --------
    tuple : (input_path, success, error_message)
    """
    try:
        img = tifffile.imread(input_path)
        
        # Handle multi-channel images
        if len(img.shape) == 3:
            if img.shape[2] <= 4:  # Likely RGB or RGBA
                img = img.mean(axis=2)
            else:  # Likely a stack
                img = img.mean(axis=0)
        
        filtered = remove_radial_artifacts_polar(
            img,
            dc_protect_radius=dc_protect,
            show_result=False,
            output_path=output_path,
            radial_percentile=percentile,
            verbose=verbose
        )
        
        return (str(input_path), True, None)
        
    except Exception as e:
        return (str(input_path), False, str(e))


def process_folder(input_folder, output_folder, dc_protect=15, percentile=50, 
                   threads=4, pattern='*.tif*', verbose=True):
    """
    Process all TIFF files in a folder using parallel threads.
    
    Parameters:
    -----------
    input_folder : str or Path
        Input folder containing TIFF files
    output_folder : str or Path
        Output folder for filtered images
    dc_protect : int
        DC protection radius
    percentile : float
        Normalization percentile
    threads : int
        Number of parallel threads
    pattern : str
        File pattern to match (default: '*.tif*')
    verbose : bool
        Print progress information
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    
    # Create output folder if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Find all matching files
    input_files = sorted(input_folder.glob(pattern))
    
    if not input_files:
        print(f"No files matching '{pattern}' found in {input_folder}")
        return
    
    print(f"Found {len(input_files)} files to process")
    print(f"Using {threads} threads")
    
    # Prepare file pairs
    file_pairs = []
    for input_path in input_files:
        output_path = output_folder / input_path.name
        file_pairs.append((input_path, output_path))
    
    # Process files in parallel
    successful = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=threads) as executor:
        # Submit all tasks
        futures = {
            executor.submit(
                process_single_file, 
                inp, 
                out, 
                dc_protect, 
                percentile,
                verbose=False
            ): (inp, out) 
            for inp, out in file_pairs
        }
        
        # Process results with progress bar
        with tqdm(total=len(file_pairs), desc="Processing") as pbar:
            for future in as_completed(futures):
                input_path, success, error = future.result()
                
                if success:
                    successful += 1
                    pbar.set_postfix_str(f"✓ {successful} | ✗ {failed}")
                else:
                    failed += 1
                    pbar.set_postfix_str(f"✓ {successful} | ✗ {failed}")
                    if verbose:
                        print(f"\nError processing {input_path}: {error}")
                
                pbar.update(1)
    
    print(f"\nProcessing complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")


def main():
    parser = argparse.ArgumentParser(
        description='Remove radial artifacts using polar coordinate normalization.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single file
  %(prog)s input.tiff --output filtered.tiff
  
  # Process folder with 4 threads
  %(prog)s input_folder/ --output output_folder/ --threads 4
  
  # Process folder with custom pattern
  %(prog)s input_folder/ --output output_folder/ --pattern "*.tiff" --threads 8
        """
    )
    parser.add_argument('input', help='Input TIFF file or folder')
    parser.add_argument('--output', '-o', required=True,
                       help='Output TIFF file or folder')
    parser.add_argument('--dc-protect', type=int, default=15,
                       help='Radius to protect around DC (default: 15)')
    parser.add_argument('--percentile', type=float, default=50,
                       help='Percentile for normalization (default: 50)')
    parser.add_argument('--threads', '-t', type=int, default=4,
                       help='Number of parallel threads for folder processing (default: 4)')
    parser.add_argument('--pattern', type=str, default='*.tif*',
                       help='File pattern for folder processing (default: *.tif*)')
    parser.add_argument('--show', action='store_true',
                       help='Display results (single file only)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    # Check if input is a file or folder
    if input_path.is_file():
        # Single file processing
        print(f"Loading: {args.input}")
        img = tifffile.imread(args.input)
        
        if len(img.shape) == 3:
            if img.shape[2] <= 4:
                img = img.mean(axis=2)
            else:
                img = img.mean(axis=0)
        
        filtered = remove_radial_artifacts_polar(
            img,
            dc_protect_radius=args.dc_protect,
            show_result=args.show,
            output_path=args.output,
            radial_percentile=args.percentile,
            verbose=True
        )
        
        print("Done!")
        
    elif input_path.is_dir():
        # Folder processing
        process_folder(
            input_path,
            args.output,
            dc_protect=args.dc_protect,
            percentile=args.percentile,
            threads=args.threads,
            pattern=args.pattern,
            verbose=args.verbose
        )
    else:
        print(f"Error: {args.input} is not a valid file or directory")
        exit(1)
        
        
if __name__ == '__main__':
    main()        
