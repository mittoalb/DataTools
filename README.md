# DataTools

Collection of conversion, reconstruction and analysis tools for scientific imaging data (synchrotron tomography, microscopy). Most utilities are exposed as command-line entry points after installation.

## Branches

- **`dev_zarr3`** (this branch) — `tiff2zarr` writes **Zarr v3** (chunks + optional sharding, OME-NGFF multiscale).
- **`dev_zarr2`** — `tiff2zarr` writes **Zarr v2** (legacy, for tools that don't speak v3 yet).

Pick the branch that matches the Zarr format you need; both branches expose the same `tiff2zarr` command name.

## Installation

```bash
pip install .          # standard
pip install -e .       # development
```

Requires Python ≥ 3.7 and `zarr>=3.0`. Other core dependencies: `numpy`, `tifffile`, `scikit-image`, `scipy`, `fabio`, `click`, `numcodecs`, `h5py`, `xraylib`, `tqdm`, `fsspec`. A few scripts pull extras at runtime: `Pillow` ([bmp2zarr](DataTools/DataFormats/bmp2zarr.py)), `google-cloud-storage` ([Gdown.py](DataTools/Google/Gdown.py)), `z5py` ([n5toslice.py](DataTools/Globus/n5toslice.py)), `tomopy` + `cupy` + `dxchange` ([tomopy_reco.py](DataTools/Tools/tomopy_reco.py)).

---

## Command-line tools

The following entry points are registered in [pyproject.toml](pyproject.toml) and become available on `$PATH` after installation.

### `tiff2zarr` — TIFF stack → Zarr v3 (multiscale OME-NGFF)

Source: [DataTools/DataFormats/tiff2zarr.py](DataTools/DataFormats/tiff2zarr.py)

Converts a folder of TIFF slices into a chunked, compressed, multi-resolution Zarr v3 store with OME-NGFF `multiscales` metadata. Builds up to 6 pyramid levels by 2× downsampling. Auto-computes intensity range (percentile or cumulative histogram), or accepts manual `min/max`. Optional Zarr v3 sharding.

```bash
tiff2zarr INPUT_DIR OUTPUT.zarr \
    --dtype uint16 \
    --chunks 64 64 64 \
    --compression zstd \
    --pixel_size 1.0 \
    --chunk_size 64 \
    --min_percentile 0.5 --max_percentile 99.5 \
    [--use_histogram] [--sample_ratio 100] \
    [--min_val 0 --max_val 65535] \
    [--shard_size 256 256 256] \
    [--verbose]
```

Key options: `--dtype` (`int8|int16|int32|uint8|uint16|float32|float64`), `--compression` (`blosclz|lz4|lz4hc|zlib|zstd`), `--use_histogram` to switch from percentile to cumulative-histogram range, `--shard_size` to enable Zarr v3 sharding.

### `tiff2zarr_v2` — TIFF stack → Zarr v2 (kept for compatibility)

Source: [DataTools/DataFormats/tiff2zarr_v2.py](DataTools/DataFormats/tiff2zarr_v2.py)

Same workflow as `tiff2zarr` but writes the **Zarr v2** format (uses `numcodecs.Blosc`, no sharding). CLI flags mirror `tiff2zarr` except `--shard_size`, which is accepted but ignored.

> If you only need v2, prefer the **`dev_zarr2`** branch where `tiff2zarr` itself writes v2 and there's no version duplication.

```bash
tiff2zarr_v2 INPUT_DIR OUTPUT.zarr --dtype uint8 --chunks 64 64 64 --compression zstd
```

### `bmp2zarr` — BMP stack → Zarr

Source: [DataTools/DataFormats/bmp2zarr.py](DataTools/DataFormats/bmp2zarr.py)

Converts a folder of BMP images to a chunked Zarr store with optional cropping and 2D/3D downsampling.

```bash
bmp2zarr INPUT_DIR OUTPUT.zarr \
    --dtype uint8 \
    --chunks 64 64 64 \
    --compression blosclz \
    --pixel_size 1.0 \
    [--crop startx:endx:starty:endy] \
    [--autocrop --autocrop-threshold 10 --pad 0] \
    [--downsample-mode 2d|3d]
```

`--autocrop` builds a max-projection across the stack, thresholds it, and crops the bounding box of foreground pixels (with optional `--pad`).

### `zarr2tiff` — Zarr resolution layer → TIFF slices

Source: [DataTools/DataFormats/zarr2tiff.py](DataTools/DataFormats/zarr2tiff.py)

Exports one resolution layer of a 3D Zarr store as a stack of per-slice TIFF files, written in parallel with a thread pool.

```bash
zarr2tiff INPUT.zarr OUTPUT_DIR \
    --resolution 0 \
    --dtype float32 \
    --threads 4
```

Output files are named `<zarr-basename>_<index:04d>.tiff`.

### `edf2aps` — ESRF EDF projections → APS-style HDF5

Source: [DataTools/Facilities/edf2aps.py](DataTools/Facilities/edf2aps.py)

Bundles a folder of EDF projections, dark fields (`darkHST*`) and flat fields (`refHST*`) into a single HDF5 file with the APS `/exchange` layout (`data`, `data_dark`, `data_white`, `theta`). The angle vector is generated from the number of projections and an angular range.

```bash
edf2aps --input_path /path/to/edf_dir \
        --output_path output.h5 \
        --arange 180 \
        [--numberofdark 20]
```

### `esrf2aps` — ESRF NXtomo HDF5 → APS-style HDF5 (virtual datasets)

Source: [DataTools/Facilities/esrf2aps.py](DataTools/Facilities/esrf2aps.py)

Re-maps an ESRF NXtomo file (multi-scan with `projections`/`flat`/`dark` titles) into the APS `/exchange` layout using **virtual datasets**, so no data is duplicated. Optionally splices flat/dark fields from a separate reference scan.

```bash
esrf2aps -i input_nxtomo.h5 -o aps.h5 [-r reference.h5]
```

### `abscalc` — Linear X-ray attenuation coefficient

Source: [DataTools/Physics/abscalc.py](DataTools/Physics/abscalc.py)

Computes the linear attenuation coefficient (cm⁻¹) of a compound at a given X-ray energy using `xraylib`. Accepts a chemical formula and density.

```bash
abscalc --formula H2O --density 1.0 --energy 25.0
# → Linear attenuation coefficient for H2O at 25.0 keV with density 1.0 g/cm^3 is: 0.508... cm^-1
```

### `create_vol` — Glue multi-folder TIFF reconstructions into a single volume

Source: [DataTools/Tools/create_vol.py](DataTools/Tools/create_vol.py)

Combines per-scan reconstructions sitting in adjacent sub-folders into a single, contiguously-numbered TIFF stack. Keeps the head of the first folder, the central N slices of each middle folder, and the tail of the last folder. Can additionally emit 8- or 16-bit binned copies.

```bash
create_vol \
    --source_root /data/run001 \
    --dest_dir    /data/run001/VOL \
    --central     1030 \
    --prefix      recon \
    [--copy] \
    [--binning 1 2 4 8 16] [--format 8|16] \
    [--start_vol 0 --end_vol 100000]
```

Without `--copy`, output files are symlinks to the originals.

### `extract_meta` — Dump HDF5 attributes and dataset values to text

Source: [DataTools/Tools/extract_meta.py](DataTools/Tools/extract_meta.py)

Walks an HDF5 file recursively and writes every attribute and dataset value to a `.txt` file with the same base name. Useful for quick inspection without opening the file in a viewer.

```bash
extract_meta path/to/file.h5
# → writes path/to/file.txt
```

### `polar` — Remove radial FFT line artifacts (polar normalization)

Source: [DataTools/Tools/polar_removal.py](DataTools/Tools/polar_removal.py)

Removes radial streak/line artifacts by computing the FFT, normalizing magnitudes in polar coordinates (per radius), and inverse-transforming back. Works on a single TIFF or a whole folder in parallel.

```bash
# Single file
polar input.tiff -o filtered.tiff [--dc-protect 15] [--percentile 50] [--show] [-v]

# Folder, 8 threads
polar input_folder/ -o output_folder/ --threads 8 --pattern "*.tif*"
```

---

## Viewing the output

A Zarr store written by `tiff2zarr` is a directory you can open from Python or from a viewer.

### From Python

```python
import zarr
g = zarr.open_group('/path/to/out.zarr', mode='r')
print(list(g.keys()))            # pyramid levels: ['0', '1', '2', ...]
print(g.attrs.get('multiscales'))  # OME-NGFF metadata
vol = g['0'][:]                  # load full-res as numpy
slice_z = g['0'][100, :, :]      # single Z-slice (lazy)
```

Quick sanity check that the pyramid is real:

```bash
python -c "
import zarr
g = zarr.open_group('/path/to/out.zarr', 'r')
for k in sorted(g.keys()): print(k, g[k].shape, g[k].dtype)
"
```

### With napari (best for OME-NGFF multiscale)

```bash
pip install "napari[all]" napari-ome-zarr
napari /path/to/out.zarr
```

To confirm napari loaded the pyramid as a single multiscale layer (not separate layers per level), open the napari console (`View → Toggle Console`) and run:

```python
viewer.layers[0].multiscale            # → True
[a.shape for a in viewer.layers[0].data]  # → [(N,Y,X), (N/2,Y/2,X/2), ...]
```

If `multiscale` is `False`, force the OME-NGFF reader:

```bash
napari --plugin napari-ome-zarr /path/to/out.zarr
```

### Format identification

- **Zarr v3** (this branch): root contains `zarr.json` with `"zarr_format": 3`.
- **Zarr v2** (`dev_zarr2` branch): root contains `.zgroup` with `{"zarr_format": 2}`.

Both formats are read transparently by `zarr.open_group` and napari.

### Round-trip back to TIFF

```bash
zarr2tiff /path/to/out.zarr /path/to/tiff_out --resolution 0 --dtype uint16 --threads 4
```

---

## Other scripts (not registered as console entry points)

These ship in the package but are invoked directly with `python -m` or as a file.

### `DataTools/Globus/n5toslice.py` — Extract a Z-slice from an N5 volume

Reads a 3D N5 dataset with `z5py` and saves a single Z-slice as a TIFF (optionally previewing it with matplotlib). Requires `z5py` and `Pillow`.

```bash
python -m DataTools.Globus.n5toslice \
    --n5-file-path P-1C_290_rec.n5 \
    --resolution 0 \
    --z-index 50 \
    --output slice_50.tiff \
    [--no-show]
```

### `DataTools/Globus/io.py` — Remote Zarr access helpers (library)

Helpers for opening remote Zarr stores over HTTPS with an optional bearer token (`open_zarr_store`), enumerating resolutions, and loading a single 2D slice (`Gload_zarr`). Used as a library, not from the command line; the `Notebooks/GlobusReader.ipynb` notebook is the reference consumer.

### `DataTools/Google/Gdown.py` — Bulk download a Google Cloud Storage prefix

Mirrors all objects under a GCS prefix to a local directory. Uses an anonymous client by default (public buckets); pass `--no-anonymous` to use ambient credentials (`gcloud auth application-default login`). Requires `google-cloud-storage`.

```bash
python DataTools/Google/Gdown.py \
    --bucket-name my-public-bucket \
    --prefix path/to/data \
    --local-dest ./downloads \
    [--no-anonymous]
```

### `DataTools/Tools/tomopy_reco.py` — Tomopy + CuPy reconstruction pipeline

End-to-end example pipeline: read APS-format HDF5, normalize, log-correct, Fourier-wavelet stripe removal, optional Paganin phase retrieval (CuPy), 360° padding, gridrec reconstruction, circular mask, write per-slice TIFFs. The file paths and parameters at the bottom of the script are hard-coded — **edit before running**. Requires `tomopy`, `dxchange`, `cupy`.

### `DataTools/Bash_scripts/chunker.sh` — Split a projection range into N reconstruction chunks

Splits `<num_projections>` evenly into `<num_chunks>` and runs the given command with `--start-proj`/`--end-proj` on each chunk, moving the `_rec` output into `_rec/_rec_<i>` between calls.

```bash
./DataTools/Bash_scripts/chunker.sh <num_chunks> <num_projections> <command...>
# Example:
./DataTools/Bash_scripts/chunker.sh 4 1500 tomocupy recon --file-name APS.h5 ...
```

### `DataTools/Bash_scripts/run_stack_reco.sh` — Batch ESRF→APS conversion + tomocupy reconstruction

Walks a directory of scans, converts each NXtomo HDF5 to APS format, and triggers `tomocupy recon`.

```bash
./DataTools/Bash_scripts/run_stack_reco.sh <data_path> <COR> <double_fov|normal> <full|try>
```

Note: the script currently invokes `ESRF2APS.py` via a relative path — adjust to call the installed `esrf2aps` entry point if you need it portable.

---

## Project layout

```
DataTools/
├── DataFormats/       tiff2zarr, tiff2zarr_v2, bmp2zarr, zarr2tiff, utils, log
├── Facilities/        edf2aps, esrf2aps           (facility format conversions)
├── Physics/           abscalc                     (X-ray attenuation)
├── Tools/             create_vol, extract_meta, polar_removal, tomopy_reco
├── Globus/            io, n5toslice               (remote/N5 access)
├── Google/            Gdown                       (GCS download)
├── Bash_scripts/      chunker.sh, run_stack_reco.sh
└── Notebooks/         GlobusReader.ipynb
```

## License

MIT — see [LICENCE](LICENCE).

## Author

**Alberto Mittone** — `amittone@anl.gov` ([@mittoalb](https://github.com/mittoalb))
