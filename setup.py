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
    description="A Keras implementation of Char-CNN: Character-level Convolutional Networks for Text Classification, Zhang et al, 2016",
    keywords="keras character char cnn nlp deep-learning",
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
    version="0.0.3"
)
