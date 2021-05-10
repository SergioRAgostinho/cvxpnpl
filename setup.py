import ast
from ast import NodeVisitor
from setuptools import setup


class VersionExtractor(NodeVisitor):
    def __init__(self):
        super().__init__()
        self.version = None

    def visit_Assign(self, node):
        if hasattr(node.targets[0], "id") and node.targets[0].id == "__version__":
            self.version = node.value.s


def parse_version():

    with open("cvxpnpl.py", "r") as f:
        content = f.read()

    tree = ast.parse(content)
    visitor = VersionExtractor()
    visitor.visit(tree)
    return visitor.version


def long_description():
    with open("README.md", "r") as f:
        return f.read()


setup(
    name="cvxpnpl",
    version=parse_version(),
    license="Apache 2.0",
    description="A convex Perspective-n-Points-and-Lines method.",
    long_description=long_description(),
    long_description_content_type="text/markdown",
    install_requires=["numpy", "scipy", "scs>=2.0.0"],
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
