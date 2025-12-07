Command-Line Tools
==================

DataTools provides several command-line utilities for data conversion and processing.
All tools are installed automatically when you install the DataTools package.

Data Format Conversion Tools
-----------------------------

tiff2zarr
~~~~~~~~~

Convert TIFF files to Zarr format.

**Syntax**::

    tiff2zarr [OPTIONS] INPUT OUTPUT

**Arguments:**

* ``INPUT``: Path to input TIFF file or directory
* ``OUTPUT``: Path to output Zarr file

**Options:**

* ``--compression TEXT``: Compression codec (default: zstd)
* ``--chunks INTEGER``: Chunk size
* ``--help``: Show help message

**Examples**::

    # Convert single TIFF
    tiff2zarr input.tiff output.zarr

    # Convert with custom compression
    tiff2zarr --compression blosc input.tiff output.zarr

    # Convert with custom chunk size
    tiff2zarr --chunks 256 input.tiff output.zarr

bmp2zarr
~~~~~~~~

Convert BMP files to Zarr format.

**Syntax**::

    bmp2zarr [OPTIONS] INPUT OUTPUT

**Arguments:**

* ``INPUT``: Path to input BMP file or directory
* ``OUTPUT``: Path to output Zarr file

**Options:**

* ``--compression TEXT``: Compression codec (default: zstd)
* ``--chunks INTEGER``: Chunk size
* ``--help``: Show help message

**Examples**::

    # Convert BMP directory
    bmp2zarr images/ output.zarr

    # Convert with custom options
    bmp2zarr --compression blosc --chunks 128 images/ output.zarr

zarr2tiff
~~~~~~~~~

Convert Zarr arrays to TIFF format.

**Syntax**::

    zarr2tiff [OPTIONS] INPUT OUTPUT

**Arguments:**

* ``INPUT``: Path to input Zarr file
* ``OUTPUT``: Path to output TIFF file

**Options:**

* ``--bigtiff``: Use BigTIFF format for large files
* ``--help``: Show help message

**Examples**::

    # Convert Zarr to TIFF
    zarr2tiff input.zarr output.tiff

    # Use BigTIFF for large files
    zarr2tiff --bigtiff input.zarr output.tiff

Facility Conversion Tools
--------------------------

esrf2aps
~~~~~~~~

Convert ESRF data format to APS format.

**Syntax**::

    esrf2aps [OPTIONS] INPUT OUTPUT

**Arguments:**

* ``INPUT``: Path to ESRF data file
* ``OUTPUT``: Path to APS output file

**Examples**::

    # Convert ESRF to APS
    esrf2aps esrf_data.edf aps_data.h5

edf2aps
~~~~~~~

Convert EDF (ESRF Data Format) files to APS format.

**Syntax**::

    edf2aps [OPTIONS] INPUT OUTPUT

**Arguments:**

* ``INPUT``: Path to EDF file
* ``OUTPUT``: Path to APS output file

**Examples**::

    # Convert EDF to APS
    edf2aps scan001.edf scan001_aps.h5

Physics Tools
-------------

abscalc
~~~~~~~

Interactive X-ray absorption calculator.

**Syntax**::

    abscalc [OPTIONS]

**Description:**

Launches an interactive calculator for X-ray absorption coefficients.
Uses xraylib for accurate calculations based on element, energy, and density.

**Examples**::

    # Launch interactive calculator
    abscalc

**Interactive prompts:**

1. Element or compound formula (e.g., Fe, H2O, SiO2)
2. X-ray energy in keV
3. Density in g/cm³ (optional, uses standard density if not provided)

Data Processing Tools
---------------------

create_vol
~~~~~~~~~~

Create volumetric datasets from image stacks.

**Syntax**::

    create_vol [OPTIONS] INPUT OUTPUT

**Arguments:**

* ``INPUT``: Path to input image stack directory
* ``OUTPUT``: Path to output volume file

**Options:**

* ``--format TEXT``: Output format (h5, zarr)
* ``--help``: Show help message

**Examples**::

    # Create HDF5 volume
    create_vol image_stack/ volume.h5

    # Create Zarr volume
    create_vol --format zarr image_stack/ volume.zarr

extract_meta
~~~~~~~~~~~~

Extract metadata from image files.

**Syntax**::

    extract_meta [OPTIONS] INPUT

**Arguments:**

* ``INPUT``: Path to input image file

**Options:**

* ``--output TEXT``: Output file for metadata (JSON format)
* ``--help``: Show help message

**Examples**::

    # Print metadata to console
    extract_meta image.tiff

    # Save metadata to file
    extract_meta --output metadata.json image.tiff

polar
~~~~~

Remove polar artifacts from tomographic reconstructions.

**Syntax**::

    polar [OPTIONS] INPUT OUTPUT

**Arguments:**

* ``INPUT``: Path to input reconstruction file
* ``OUTPUT``: Path to output file

**Options:**

* ``--method TEXT``: Removal method (default: auto)
* ``--help``: Show help message

**Examples**::

    # Remove polar artifacts
    polar reconstruction.h5 cleaned.h5

    # Use specific method
    polar --method interpolation reconstruction.h5 cleaned.h5

Common Workflow Examples
------------------------

Complete Processing Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Process a tomography dataset from raw TIFF to cleaned volume::

    # Step 1: Convert TIFF stack to Zarr
    tiff2zarr projections/ raw_data.zarr

    # Step 2: Create volume
    create_vol raw_data.zarr volume.zarr

    # Step 3: Remove artifacts
    polar volume.zarr cleaned_volume.zarr

    # Step 4: Convert back to TIFF if needed
    zarr2tiff cleaned_volume.zarr final_volume.tiff

Multi-Facility Workflow
~~~~~~~~~~~~~~~~~~~~~~~

Process data from ESRF for use at APS::

    # Convert ESRF data
    esrf2aps esrf_scan.edf aps_scan.h5

    # Convert to Zarr for efficient processing
    tiff2zarr aps_scan.h5 aps_scan.zarr

    # Extract metadata
    extract_meta aps_scan.zarr --output scan_metadata.json

Batch Processing
~~~~~~~~~~~~~~~~

Process multiple files using shell scripting::

    # Bash example: Convert all TIFF files in a directory
    for file in *.tiff; do
        tiff2zarr "$file" "${file%.tiff}.zarr"
    done

    # Process all Zarr files
    for file in *.zarr; do
        polar "$file" "cleaned_${file}"
    done

Performance Tips
----------------

1. **Use appropriate chunk sizes**: Larger chunks = better compression, smaller chunks = better random access
2. **Choose compression wisely**: zstd offers good balance of speed and compression
3. **Parallel processing**: Many tools support parallel processing through environment variables
4. **Memory management**: Use chunked formats (Zarr) for large datasets

Getting Help
------------

All command-line tools support the ``--help`` flag for detailed usage information::

    tiff2zarr --help
    bmp2zarr --help
    abscalc --help

For more information, see the API documentation for each module.
