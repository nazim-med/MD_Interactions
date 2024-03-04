from setuptools import setup, find_packages

setup(
    name='MD_PLIP',
    version='0.2',
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "rdkit",
        "mdtraj",
        "matplotlib",
        "scikit-learn"
    ],
    entry_points={
        'console_scripts': [
            'md_plip=MD_PLIP.MD_PLIP_cmd:main',
        ],
    },
)