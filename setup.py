# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='MuSiCal',
    version='0.1.0',
    description='General tools for mutational signature analysis',
    long_description=readme,
    author='Hu Jin',
    author_email='hu_jin@hms.harvard.edu',
    url='https://github.com/Hu-JIN/MuSiCal',
    license=license,
    packages=find_packages(exclude=('tests', 'docs', 'benchmarks',
                                    'examples', 'dev'))
)
