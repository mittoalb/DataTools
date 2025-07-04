from setuptools import setup, find_packages

setup(
    name="datatools",
    version="0.0.1",
    description="A package for data conversion and analysis tools",
    author="Alberto Mittone",
    author_email="amittone@anl.gov",
    url="https://github.com/mittoalb/Allen",
    packages=find_packages(),  # Automatically find subpackages
    install_requires=["fabio","zarr","click","numcodecs","h5py","xraylib","tqdm"],
    entry_points={
        'console_scripts': [
            'tiff2zarr=DataTools.DataFormats.tiff2zarr:main',
            'tiff2zarrOPT=DataTools.DataFormats.tiff2zarr_opt:main',            
            'zarr2tiff=DataTools.DataFormats.zarr2tiff:main',
            'tiff2omezarr=DataTools.DataFormats.tiff2omezarr:main',
            'esrf2aps=DataTools.Facilities.esrf2aps:main',
            'edf2aps=DataTools.Facilities.edf2aps:main',
            'abscalc=DataTools.Physics.abscalc:main',
            'create_vol=DataTools.Tools.create_vol:main',
            'extract_meta=DataTools.Tools.extract_meta:main'     
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

