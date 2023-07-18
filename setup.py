import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "DutchScaler",
    version = "0.0.1",
    author = "Etienne van de Bijl, Jan Klein, Joris Pries",
    author_email = "evdb@cwi.nl",
    description = "Binary Performance Quantification",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/etiennevandebijl/Dutch-Scaler",
    packages = setuptools.find_packages(),
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)