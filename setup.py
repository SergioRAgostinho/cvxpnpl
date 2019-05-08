from setuptools import setup


def install_requires():
    """Generate list with dependency requirements"""

    deps = []
    with open("requirements.txt", "r") as f:
        for line in f:
            deps.append(line[:-1])
    return deps

def long_description():
    with open("README.md", "r") as f:
        return f.read()



setup(
    name="cvxpnpl",
    version="0.1.1",
    license="Apache 2.0",
    description="A convex Perspective-n-Points-and-Lines method.",
    long_description=long_description(),
    long_description_content_type="text/markdown",
    install_requires=install_requires(),
    author="Sérgio Agostinho",
    author_email="sergio@sergioagostinho.com",
    url="https://github.com/SergioRAgostinho/cvxpnpl",
    py_modules=["cvxpnpl"],
    python_requires=">=3.5",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering",
    ],
)
