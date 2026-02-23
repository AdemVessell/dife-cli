from setuptools import setup, find_packages
setup(
    name="dife-cli",
    version="0.2.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    entry_points={"console_scripts": ["dife=dife_cli.cli:main"]},
    install_requires=["numpy", "scipy", "pandas", "matplotlib", "torch"],
    python_requires=">=3.9",
)
