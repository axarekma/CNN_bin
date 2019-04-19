import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cnnbin",
    version="0.0.2",
    author="Axel Ekman",
    author_email="axel.ekman@iki.fi",
    description="image binning with CNN filtering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/axarekma/CNN_bin",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
