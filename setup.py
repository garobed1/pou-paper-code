import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="garo-rpi-graduate-work",
    version="0.1.0",
    author="Garo Bedonian",
    author_email="bedong@rpi.edu",
    description="A repository for the aerostructural multidisciplinary optimization-under-uncertainty code developed during my PhD.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/garobed1/garo-rpi-graduate-work",
    packages=[
        'beam',
        'functions',
        'infill',
        'meshes',
        'mphys_comp',
        'old_components',
        'optimization',
        'scratch',
        'surrogate',
        'utils'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[]
)