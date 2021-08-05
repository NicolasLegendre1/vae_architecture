"""Create instructions to build the ioSPI package."""

import setuptools

requirements = []

setuptools.setup(
    name="vae_architecture",
    maintainer="Nicolas Legendre",
    version="0.0.1",
    maintainer_email="nicolas.legendre@student-cs.fr",
    description="VAE tools",
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/NicolasLegendre1/vae_architecture",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    zip_safe=False,
)
