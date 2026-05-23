
setup(
    name="delta-jepa",
    version="1.0.0",
    author="The Bridge Architect",
    author_email="bigwiginfohub@gmail.com",
    description="Auditable world models with governance layers",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/bigwiginfohub-wq/delta-jepa",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.21.0",
        "einops>=0.6.0",
    ],
)