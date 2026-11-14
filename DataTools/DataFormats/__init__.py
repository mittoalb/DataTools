# The tiff2zarr / zarr2tiff / bmp2zarr CLI entrypoints are NOT auto-imported
# here to keep this package importable in environments that don't have zarr
# (or have the "wrong" major version). Import them explicitly when needed:
#   from DataTools.DataFormats import tiff2zarr
# All three handle both Zarr v2 and v3 output via a --zarr-format flag; the
# chosen format is also recorded in root.attrs["zarr_format"] in the store.
from .utils import calculate_global_min_max, load_tiff_chunked, downsample, minmaxHisto

