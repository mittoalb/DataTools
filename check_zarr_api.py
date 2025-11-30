#!/usr/bin/env python3
"""
Quick script to check zarr v3 API structure
"""
import zarr
print(f"Zarr version: {zarr.__version__}")
print()

# Check what's in zarr.codecs
try:
    import zarr.codecs as codecs
    print("Available in zarr.codecs:")
    print([name for name in dir(codecs) if not name.startswith('_')])
    print()
except ImportError as e:
    print(f"Cannot import zarr.codecs: {e}")
    print()

# Check what's in zarr.store or zarr.storage
for store_module in ['zarr.store', 'zarr.storage']:
    try:
        store_mod = __import__(store_module, fromlist=[''])
        print(f"Available in {store_module}:")
        print([name for name in dir(store_mod) if not name.startswith('_')])
        print()
    except ImportError as e:
        print(f"Cannot import {store_module}: {e}")
        print()

# Try to create a simple zarr v3 array to see what works
print("Testing basic zarr v3 array creation:")
try:
    import tempfile
    import shutil
    tmpdir = tempfile.mkdtemp()

    # Try different approaches
    try:
        from zarr.storage import LocalStore
        print("✓ LocalStore imported from zarr.storage")
    except:
        try:
            from zarr.store import LocalStore
            print("✓ LocalStore imported from zarr.store")
        except Exception as e:
            print(f"✗ Cannot import LocalStore: {e}")

    shutil.rmtree(tmpdir)
except Exception as e:
    print(f"Error in test: {e}")
