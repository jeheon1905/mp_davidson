import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mp_davidson",
    version="0.0.1",
    author="Jeheon Woo",
    author_email="jhwoo1905@gmail.com",
    description="Mixed-precision for Davidson",
    long_description=long_description,
    url="https://github.com/jeheon1905/mp_davidson",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.7",
    # install_requires -> pip install -r requirements.txt
)
