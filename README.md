# DataTools

Collections of tools for data conversion and management

## Overview

DataTools is a Python package designed for scientific data processing, with a focus on converting imaging files from various formats into efficient, modern storage formats. The package specializes in handling scientific imaging data such as EDF (European Data Format) and TIFF files, converting them to optimized formats like HDF5 and Zarr for improved performance and storage efficiency.

## Features

- **Multi-format Support**: Convert between various imaging file formats including:
  - EDF (European Data Format)
  - TIFF (Tagged Image File Format)
  - HDF5 (Hierarchical Data Format version 5)
  - Zarr (cloud-optimized n-dimensional arrays)

- **Scientific Data Processing**: Optimized for handling large-scale scientific imaging datasets

- **Efficient Storage**: Convert legacy formats to modern, compressed, and chunked storage formats for better I/O performance

- **Data Management**: Tools for organizing and managing large collections of imaging data

## Installation

### From Source

To install DataTools from source, navigate to the project root directory and run:

```bash
pip install .
```

### Development Installation

For development purposes, install in editable mode:

```bash
pip install -e .
```

## Requirements

- Python 3.7+
- NumPy
- h5py (for HDF5 support)
- zarr (for Zarr support)
- Additional dependencies as specified in `requirements.txt` or `setup.py`

## Usage

### Basic File Conversion

```python
from datatools import convert_imaging_data

# Convert EDF to HDF5
convert_imaging_data('input.edf', 'output.h5', format='hdf5')

# Convert TIFF to Zarr
convert_imaging_data('input.tiff', 'output.zarr', format='zarr')
```

### Batch Processing

```python
from datatools import batch_convert

# Process multiple files
input_files = ['file1.edf', 'file2.edf', 'file3.edf']
batch_convert(input_files, output_dir='converted/', format='hdf5')
```

## File Format Support

### Input Formats
- **EDF**: European Data Format, commonly used in synchrotron facilities
- **TIFF**: Tagged Image File Format, widely used for imaging data
- Other scientific imaging formats (see documentation for full list)

### Output Formats
- **HDF5**: Hierarchical data format with compression and chunking support
- **Zarr**: Cloud-optimized arrays with advanced compression options

## Use Cases

- **Synchrotron Data Processing**: Convert large volumes of EDF files from synchrotron experiments
- **Microscopy Data Management**: Process and organize TIFF stacks from microscopy
- **Archive Migration**: Modernize legacy imaging datasets for better performance
- **Cloud Storage Optimization**: Convert data to cloud-friendly formats like Zarr

## Performance Benefits

- **Compression**: Modern formats support various compression algorithms
- **Chunking**: Optimized data layout for partial reading
- **Parallel I/O**: Better support for concurrent access
- **Metadata**: Enhanced metadata storage capabilities

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

### Development Setup

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
4. Install in development mode: `pip install -e .`
5. Run tests: `python -m pytest`

## Author

**Alberto Mittone** ([@mittoalb](https://github.com/mittoalb))

## Support

For questions, issues, or support, please open an issue on GitHub.
