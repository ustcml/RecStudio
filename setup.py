from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os

from setuptools import setup, find_packages

install_requires = ['numpy>=1.20.1', 'torch>=1.9.0', 'scipy>=1.6.0', 'pandas>=1.3.0', 'tqdm>=4.48.2',
                    'colorlog==4.7.2','colorama==0.4.4', 'pyyaml>=5.1.0', 'tensorboard>=2.5.0', 
                     'faiss-gpu==1.7.2', 'torchmetrics==0.7.3']

setup_requires = []

extras_require = {}

classifiers =  ['License :: OSI Approved :: MIT License',
                'License :: OSI Approved :: MIT License',
                'Programming Language :: Python',
                'Programming Language :: Python :: 3',
                'Programming Language :: Python :: 3.8']

long_description = 'RecStudio is a modular, efficient, unified, and comprehensive recommendation library based on PyTorch.'\
                   'We divide all the models into 3 basic classes according to the number of towers: TowerFree, ItemTower, TwoTower, '\
                   'and cover models in 4 tasks: General Recommendation, Sequential Recommendation, Knowledge-based Recommendation, Social-Network-based Recommendation. '\
                   'View github page: https://github.com/ustcml/RecStudio'

# Readthedocs requires Sphinx extensions to be specified as part of
# install_requires in order to build properly.
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if on_rtd:
    install_requires.extend(setup_requires)

setup(
    name='recstudio',
    version=
    '0.0.2a1',  # please remember to edit recbole/__init__.py in response, once updating the version
    description='A modular, efficient, unified, and comprehensive recommendation library based on PyTorch.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/ustcml/RecStudio',
    author='USTCML',
    author_email='liandefu@ustc.edu.cn',
    packages=[
        package for package in find_packages()
        if package.startswith('recstudio')
    ],
    include_package_data=True,
    install_requires=install_requires,
    setup_requires=setup_requires,
    extras_require=extras_require,
    zip_safe=False,
    classifiers=classifiers,
)