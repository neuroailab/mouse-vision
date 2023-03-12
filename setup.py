from setuptools import setup, find_packages

setup(
    name="mouse_vision",
    version="1.0", 
    packages=find_packages(),
    install_requires=[
        "allensdk==2.2.0",
        "h5py==2.10.0",
        "joblib==0.17.0",
        "jsonpickle==1.4.1",
        "matplotlib==3.3.2",
        "numpy==1.18.5",
        "pandas==0.25.3",
        "Pillow==7.2.0",
        "protobuf==3.17.3",
        "pymongo==3.11.1",
        "regex==2020.11.13",
        "scikit_learn==0.24.2",
        "scipy==1.5.2",
        "Shapely==1.7.1",
        "torch==1.6",
        "torchvision==0.7.0",
        "xarray==0.15.1"
    ],
    python_requires=">=3.6"
)
