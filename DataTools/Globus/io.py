import zarr
import time
import fsspec

from ..log import info, error

def open_zarr_store(url, token=None):
    if token:
        # Use fsspec to handle the token for authentication
        fs = fsspec.filesystem('https', headers={'Authorization': f'Bearer {token}'})
        store = zarr.storage.FSStore(url, fs=fs)
    else:
        store = zarr.storage.FSStore(url)
    return zarr.open_group(store, mode='r')

def find_resolutions(store):
    metadata = store.attrs.asdict()
    multiscales = metadata.get('multiscales', [])
    if not multiscales:
        raise ValueError("No multiscales information found in the metadata.")
    datasets = multiscales[0].get('datasets', [])
    resolutions = [dataset['path'] for dataset in datasets]
    return resolutions

def load_slice(store, resolution_level, axis, index):
    if resolution_level in store:
        start_time = time.time()
        if axis == 0:
            data_slice = store[resolution_level][index, :, :]
        elif axis == 1:
            data_slice = store[resolution_level][:, index, :]
        elif axis == 2:
            data_slice = store[resolution_level][:, :, index]
        else:
            raise ValueError("Invalid axis value. Must be 0, 1, or 2.")
        end_time = time.time()
        info(f"Loaded slice at index {index} along axis {axis} from resolution {resolution_level} in {end_time - start_time:.2f} seconds")
        return data_slice
    else:
        raise ValueError(f"Resolution level {resolution_level} not found in the ZARR store.")

def validate_slice(data_slice):
    if data_slice.ndim != 2:
        raise ValueError("Loaded slice is not a 2D array")
    return data_slice

def get_volume_size(store, resolution_level):
    if resolution_level in store:
        shape = store[resolution_level].shape
        return shape
    else:
        raise ValueError(f"Resolution level {resolution_level} not found in the ZARR store.")



def Gload_zarr(zarr_url, token, resolution_layer, plane_axis, plane_index):
    """
    Load a 2D slice from a Zarr store given a URL, token, resolution layer, and slice parameters.

    Args:
        zarr_url (str): The URL to the Zarr store.
        token (str): Authentication token for the Zarr store (or empty string if not required).
        resolution_layer (int): The resolution level to access.
        plane_axis (int): Axis of the plane (0=axial, 1=sagittal, 2=coronal).
        plane_index (int): The index along the chosen axis.

    Returns:
        np.ndarray: The 2D slice data loaded from the Zarr store.
    """
    info(f"Attempting to open Zarr store at URL: {zarr_url}")

    try:
        # Open the Zarr store
        store = open_zarr_store(zarr_url, token)
        info(f"Zarr store opened successfully: {zarr_url}")
    except Exception as e:
        error(f"Failed to open Zarr store: {e}")
        raise

    try:
        # Find available resolutions
        resolutions = find_resolutions(store)
        info(f"Available resolutions: {resolutions}")

        if resolution_layer >= len(resolutions):
            error(f"Resolution layer {resolution_layer} is out of range. Available layers: {len(resolutions)}")
            raise ValueError("Invalid resolution layer index.")

        current_resolution = resolutions[resolution_layer]
        info(f"Selected resolution layer: {current_resolution}")
    except Exception as e:
        error(f"Error determining resolutions: {e}")
        raise

    try:
        # Get the volume size for the selected resolution
        volume_size = get_volume_size(store, current_resolution)
        info(f"Volume size for resolution '{current_resolution}': {volume_size}")

        # Validate the requested slice index
        if not (0 <= plane_index < volume_size[plane_axis]):
            error(
                f"Requested index {plane_index} is out of range for axis {plane_axis}. "
                f"Valid range: 0 to {volume_size[plane_axis] - 1}"
            )
            raise ValueError("Invalid slice index.")

        # Load the requested slice
        slice_data = load_slice(store, current_resolution, plane_axis, plane_index)
        info(f"Slice successfully loaded: Axis {plane_axis}, Index {plane_index}")

        # Validate the slice data
        validated_slice_data = validate_slice(slice_data)
        info(f"Slice validation successful: {validated_slice_data.shape}")

        return validated_slice_data
    except Exception as e:
        error(f"Failed to load or validate slice: {e}")
        raise
