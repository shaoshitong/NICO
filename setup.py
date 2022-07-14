from setuptools import setup, find_packages

setup(
    name='nico-mcislab840',
    version='0.0.1',
    description='2022 NICO Context Generalization Challenge',
    packages=find_packages(),
    install_requires=[
         'timm',
         'Pillow',
         'tqdm'
    ],
)