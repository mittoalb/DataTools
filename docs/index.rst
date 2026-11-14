.. DataTools documentation master file, created by
   sphinx-quickstart on Thu Mar 13 21:24:49 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to DataTools's documentation!
=====================================

**DataTools** is a Python package for scientific data processing, designed to convert
imaging files such as EDF, TIFF, and BMP into efficient formats like HDF5 and Zarr.
It provides tools for data format conversion, physics calculations, and facility-specific
data handling.

Features
--------

* **Data Format Conversion**: Convert between TIFF, BMP, EDF, Zarr, and HDF5 formats
* **Physics Calculations**: X-ray absorption calculations using xraylib
* **Facility Tools**: ESRF to APS data conversion utilities
* **Data Processing**: Volume creation, metadata extraction, and polar artifact removal
* **Cloud Integration**: Globus and Google Cloud Storage support

Quick Start
-----------

Installation::

    pip install datatools

Basic usage::

    # Convert TIFF to Zarr
    tiff2zarr input.tiff output.zarr

    # Convert BMP to Zarr
    bmp2zarr input_directory output.zarr

    # Calculate X-ray absorption
    abscalc

Documentation Contents
======================

.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   installation
   usage
   cli_tools

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api/dataformats
   api/facilities
   api/physics
   api/tools
   api/globus

.. toctree::
   :maxdepth: 1
   :caption: Development:

   contributing
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
