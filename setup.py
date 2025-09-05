from setuptools import setup, find_packages

setup(
    name="dftorch",
    version="0.1.0",
    description="A Density Functional Tight Binding (DFTB) implementation in PyTorch.",
    author="A.M.N. Niklasson, M. Kulichenko",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "numpy",
        "scipy",
        "pandas"
    ],
    python_requires=">=3.7",
    include_package_data=True,
    zip_safe=False,
)
