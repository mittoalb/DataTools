Usage Guide
===========

This guide provides examples of using DataTools for common data processing tasks.

Data Format Conversion
----------------------

TIFF to Zarr
~~~~~~~~~~~~

Convert TIFF image stacks to Zarr format for efficient storage and processing::

    from DataTools.DataFormats import tiff2zarr

    # Convert a single TIFF file
    tiff2zarr.convert('input.tiff', 'output.zarr')

Or use the command-line interface::

    tiff2zarr input.tiff output.zarr

BMP to Zarr
~~~~~~~~~~~

Convert BMP images to Zarr format::

    from DataTools.DataFormats import bmp2zarr

    # Convert BMP directory to Zarr
    bmp2zarr.convert('input_directory/', 'output.zarr')

Command-line usage::

    bmp2zarr input_directory/ output.zarr

Zarr to TIFF
~~~~~~~~~~~~

Convert Zarr arrays back to TIFF format::

    from DataTools.DataFormats import zarr2tiff

    # Convert Zarr to TIFF stack
    zarr2tiff.convert('input.zarr', 'output.tiff')

Command-line usage::

    zarr2tiff input.zarr output.tiff

Facility Data Conversion
-------------------------

ESRF to APS
~~~~~~~~~~~

Convert data from ESRF format to APS format::

    from DataTools.Facilities import esrf2aps

    # Convert ESRF data
    esrf2aps.convert('esrf_data.edf', 'aps_output.h5')

Command-line usage::

    esrf2aps esrf_data.edf aps_output.h5

EDF to APS
~~~~~~~~~~

Convert EDF files to APS format::

    from DataTools.Facilities import edf2aps

    # Convert EDF to APS
    edf2aps.convert('input.edf', 'output.h5')

Command-line usage::

    edf2aps input.edf output.h5

Physics Calculations
--------------------

X-ray Absorption
~~~~~~~~~~~~~~~~

Calculate X-ray absorption coefficients::

    from DataTools.Physics import abscalc

    # Calculate absorption for a material
    mu = abscalc.calculate_absorption(
        element='Fe',
        energy=10.0,  # keV
        density=7.87  # g/cm^3
    )

Interactive command-line calculator::

    abscalc

Data Processing Tools
---------------------

Volume Creation
~~~~~~~~~~~~~~~

Create volumetric datasets from image stacks::

    from DataTools.Tools import create_vol

    # Create volume from images
    create_vol.create('image_stack/', 'output_volume.h5')

Command-line usage::

    create_vol image_stack/ output_volume.h5

Metadata Extraction
~~~~~~~~~~~~~~~~~~~

Extract metadata from image files::

    from DataTools.Tools import extract_meta

    # Extract metadata
    metadata = extract_meta.extract('image.tiff')
    print(metadata)

Command-line usage::

    extract_meta image.tiff

Polar Artifact Removal
~~~~~~~~~~~~~~~~~~~~~~~

Remove polar artifacts from tomographic reconstructions::

    from DataTools.Tools import polar_removal

    # Remove polar artifacts
    polar_removal.process('input.h5', 'output.h5')

Command-line usage::

    polar input.h5 output.h5

Working with Zarr Arrays
-------------------------

Zarr arrays provide efficient chunked, compressed storage for large datasets.

Basic Operations
~~~~~~~~~~~~~~~~

::

    import zarr
    import numpy as np
    from DataTools.DataFormats import tiff2zarr

    # Convert TIFF to Zarr
    tiff2zarr.convert('data.tiff', 'data.zarr')

    # Open Zarr array
    z = zarr.open('data.zarr', mode='r')

    # Access data
    print(z.shape)
    print(z.dtype)

    # Read subset
    subset = z[0:10, :, :]

Compression Options
~~~~~~~~~~~~~~~~~~~

Configure compression when creating Zarr arrays::

    import zarr
    from numcodecs import Blosc

    # Create compressed Zarr array
    compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
    z = zarr.open(
        'compressed.zarr',
        mode='w',
        shape=(1000, 1000, 1000),
        chunks=(100, 100, 100),
        dtype='float32',
        compressor=compressor
    )

Cloud Storage Integration
--------------------------

Globus Transfer
~~~~~~~~~~~~~~~

Use Globus for large data transfers::

    from DataTools.Globus import io

    # Configure Globus transfer
    io.transfer('source_endpoint', 'dest_endpoint', 'file.zarr')

Best Practices
--------------

1. **Choose appropriate chunk sizes**: For 3D data, use chunks of ~1-10 MB for optimal performance
2. **Use compression**: Enable compression for Zarr arrays to reduce storage requirements
3. **Parallel processing**: Many DataTools functions support parallel processing for large datasets
4. **Memory management**: Use chunked reading/writing for datasets larger than available RAM

Example Workflows
-----------------

Complete Tomography Processing Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    # 1. Convert raw TIFF data to Zarr
    tiff2zarr input_projections/ raw_data.zarr

    # 2. Remove polar artifacts
    polar raw_data.zarr processed_data.zarr

    # 3. Create volume
    create_vol processed_data.zarr final_volume.h5

    # 4. Extract metadata
    extract_meta final_volume.h5

Multi-facility Data Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    # Convert ESRF data to APS format
    esrf2aps esrf_scan001.edf aps_scan001.h5

    # Convert to Zarr for efficient processing
    tiff2zarr aps_scan001.h5 aps_scan001.zarr

    # Process with analysis tools
    polar aps_scan001.zarr processed_scan001.zarr
