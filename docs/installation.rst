Installation
============

Requirements
------------

DataTools requires Python 3.6 or higher.

Dependencies
~~~~~~~~~~~~

The package depends on the following libraries:

* fabio - For reading scientific image formats
* zarr >= 2.0 - For Zarr format support. zarr 2.x produces v2 stores only; zarr 3.x produces both v2 and v3 (``--zarr-format`` flag). Install ``zarr>=3`` if you want the choice.
* click - Command-line interface creation
* numcodecs - Compression codecs for Zarr
* h5py - HDF5 file format support
* xraylib - X-ray physics calculations
* tqdm - Progress bars
* tifffile - TIFF file handling
* scikit-image - Image processing
* z5py - N5/Zarr file format support

Optional dependencies for cloud support:

* google-api-core
* google-auth
* google-cloud-core
* google-cloud-storage
* google-crc32c
* google-resumable-media

Install from PyPI
-----------------

To install DataTools from PyPI (once published)::

    pip install datatools

Install from Source
-------------------

To install the latest development version from source:

1. Clone the repository::

    git clone https://github.com/mittoalb/DataTools.git
    cd DataTools

2. Install in development mode::

    pip install -e .

Or install normally::

    pip install .

Install with Optional Dependencies
-----------------------------------

To install with Google Cloud Storage support::

    pip install datatools[cloud]

Development Installation
------------------------

For development, it's recommended to create a virtual environment:

Using venv::

    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -e .

Using conda::

    conda create -n datatools python=3.9
    conda activate datatools
    pip install -e .

Verifying Installation
----------------------

To verify that DataTools is installed correctly, you can:

1. Import the package in Python::

    import DataTools
    print(DataTools.__version__)

2. Check the command-line tools::

    tiff2zarr --help
    bmp2zarr --help
    abscalc --help

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**ImportError for xraylib**

If you encounter issues installing xraylib, you may need to install it separately::

    conda install -c conda-forge xraylib

**Zarr version conflicts**

DataTools works with zarr 2.x and zarr 3.x. The difference is what the writers
can produce:

* zarr 2.x: writes Zarr v2 stores only. ``--zarr-format 3`` will error out.
* zarr 3.x: writes **both** Zarr v2 and v3, selectable per-run with
  ``--zarr-format {2,3}``. Readers (``zarr2tiff``) handle both formats.

If you want the choice of output format, install zarr>=3::

    pip install --upgrade "zarr>=3"

**Permission errors on Linux/macOS**

If you encounter permission errors, use the ``--user`` flag::

    pip install --user datatools
