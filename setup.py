from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()



setup(
    name="SentiLib",
    version="0.1.0",
    author="Wilfredo Graterol",
    author_email="wgraterol64@gmail.com",
    description="Emotion recognition in images and text",
    long_description=readme,
    long_description_content_type="text/markdown",
    license="MIT License",
    url="https://github.com/WGraterol64/SentiLib",
    packages=find_packages(),
    install_requires=[
        'pandas >= 1.3.5',
        'numpy >= 1.21.5',
        'torch==1.9.0',
        'torchtext==0.10.0',
        'torchvision==0.10.0',
        'opencv-python >= 4.1.2',
        'speechrecognition == 3.8.1',
        'sentence-transformers == 2.2.0',
        'deepface >= 0.0.73',
        'rdflib >= 6.1.1'
    ],
    classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent"
    ]
)
