from setuptools import setup, find_packages

setup(
    name="qtbirds_package",
    version="1.1.0",
    description="Prepares TreePPL data",
    author="Viktor Senderov and Jan Kudlicka",
    author_email="vsenderov@gmail.com",
    packages=['qtbirds'],
    install_requires=[
        "numpy",
        "treeppl"
    ],
)