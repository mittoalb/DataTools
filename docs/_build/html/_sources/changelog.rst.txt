Changelog
=========

All notable changes to DataTools will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[Unreleased]
------------

Added
~~~~~
* Comprehensive documentation for Read the Docs
* API reference documentation for all modules
* Installation guide
* Usage examples and tutorials
* CLI tools reference
* Contributing guidelines

Changed
~~~~~~~
* Updated requirements.txt with proper dependencies
* Fixed Sphinx configuration issues
* Improved Read the Docs configuration

[0.0.1] - 2025-05-26
--------------------

Added
~~~~~
* Initial release
* TIFF to Zarr conversion (tiff2zarr)
* BMP to Zarr conversion (bmp2zarr)
* Zarr to TIFF conversion (zarr2tiff)
* ESRF to APS data conversion (esrf2aps)
* EDF to APS conversion (edf2aps)
* X-ray absorption calculator (abscalc)
* Volume creation tool (create_vol)
* Metadata extraction tool (extract_meta)
* Polar artifact removal (polar)
* Globus integration utilities
* Google Cloud Storage support
* TomoPy reconstruction utilities
* Basic documentation structure
* MIT License
* Python package structure with setuptools

Features
~~~~~~~~
* Support for multiple image formats (TIFF, BMP, EDF)
* Efficient Zarr-based storage with compression
* HDF5 file format support
* Command-line interface for all tools
* Chunk-based processing for large datasets
* Progress bars for long-running operations
* Configurable compression codecs
* Integration with xraylib for physics calculations

Dependencies
~~~~~~~~~~~~
* fabio - Scientific image format reading
* zarr >= 3.0.0 - Chunked array storage
* click - CLI framework
* numcodecs - Compression codecs
* h5py - HDF5 support
* xraylib - X-ray physics
* tqdm - Progress bars
* tifffile - TIFF file handling
* scikit-image - Image processing
* z5py - N5/Zarr support

[0.0.0] - 2025-03-13
--------------------

* Project initialization
* Repository setup
* Basic package structure
