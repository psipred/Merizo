import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="merizo",
    version="2.0",
    author="UCL Bioinformatics Group",
    author_email="psipred@cs.ucl.ac.uk",
    description="Protein domain segmentation using Merizo",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/psipred/DMPfold2",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    keywords="protein structure prediction deep learning alignment end-to-end",
    scripts=["bin/merizo"],
    install_requires=[
        "torch",
        "torchvision",
        "torchaudio",
        "matplotlib",
        "natsort",
        "rotary_embedding_torch",
    ],
    include_package_data=True,
)
