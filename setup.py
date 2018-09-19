from setuptools import setup
from setuptools import find_packages

setup(
    name='data-analysis-quickie',
    version='1.0.0',
    packages=find_packages(exclude=['tests*','sample*']),
    url='https://github.com/ggarrett13/data-analysis-quickie.git',
    license='MIT',
    author='Geoffrey Hyde Garrett',
    author_email='g.h.garrett13@gmail.com',
    description=
    'This package is purposed towards providing quick and easy data analysis tools for statistical modelling. '
)
