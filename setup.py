import setuptools


setuptools.setup(
    author="Rany Keddo",
    extras_require={
        "test": [
            "pytest"
        ]
    },
    install_requires=[
        "keras",
        "pandas",
        "numpy"
    ],
    license="MIT",
    name="char-cnn",
    package_data={
        "char-cnn": [
            "data/dbpedia/*.gz",
        ]
    },
    packages=setuptools.find_packages(
        exclude=[
            "tests"
        ]
    ),
    url="https://github.com/purzelrakete/char-cnn",
    version="0.0.2"
)
