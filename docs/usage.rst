Usage Guide
===========

This guide provides examples of using DataTools for common data processing tasks.

Data Format Conversion
----------------------

TIFF to Zarr
~~~~~~~~~~~~

Convert a *directory* of TIFF slices into a multiscale OME-Zarr store. Both
Zarr v2 and v3 output formats are supported from a single CLI; pick with
``--zarr-format``::

    # Default: Zarr v2 (readable by zarr 2.x and 3.x)
    tiff2zarr /path/to/tiffs/ /path/to/output.zarr

    # Zarr v3 (requires zarr>=3)
    tiff2zarr /path/to/tiffs/ /path/to/output.zarr --zarr-format 3

The number of pyramid levels is picked automatically from the input dims; the
intensity range is auto-calibrated from percentiles. See :doc:`cli_tools` for
the full option list.

To tell which format a store is afterwards::

    python -c 'import zarr; g=zarr.open("output.zarr"); print(dict(g.attrs))'

Look for the ``zarr_format`` entry (2 or 3). On disk, v2 stores have
``.zgroup``/``.zarray`` metadata files; v3 stores have ``zarr.json``.

BMP to Zarr
~~~~~~~~~~~

Convert a directory of BMP images into a multiscale Zarr store (same
``--zarr-format`` flag as tiff2zarr)::

    bmp2zarr /path/to/bmps/ /path/to/output.zarr --zarr-format 2

Supports optional cropping (``--crop startx:endx:starty:endy``) and
``--autocrop`` based on a thresholded max projection.

Zarr to TIFF
~~~~~~~~~~~~

Read back a chosen resolution layer of a Zarr store and write it as individual
TIFF slices (works on both v2 and v3 stores — the reader auto-detects)::

    zarr2tiff /path/to/input.zarr /path/to/tiff_output_dir/

Pick a non-base resolution with ``--resolution 2`` for a downsampled export.

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

    import subprocess
    import zarr

    # Convert a directory of TIFFs to Zarr via the CLI
    subprocess.run([
        'tiff2zarr', 'tiffs/', 'data.zarr', '--zarr-format', '3',
    ], check=True)

    # Open the resulting store (zarr>=3 auto-detects v2 vs v3)
    z = zarr.open('data.zarr', mode='r')
    print(z['0'].shape, z['0'].dtype)
    print(dict(z.attrs))   # includes "zarr_format": 2 or 3

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

    # 1. Convert raw TIFF data to Zarr (v3; drop the flag for v2)
    tiff2zarr input_projections/ raw_data.zarr --zarr-format 3

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

    # Convert the resulting TIFF stack to Zarr for efficient processing
    tiff2zarr aps_scan001_tiffs/ aps_scan001.zarr --zarr-format 3

    # Process with analysis tools
    polar aps_scan001.zarr processed_scan001.zarr
