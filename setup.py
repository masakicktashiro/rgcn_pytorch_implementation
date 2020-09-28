from setuptools import setup
from setuptools import find_packages

setup(name='rgcn_torch',
      version='0.0.1',
      description='Graph Convolutional Networks for (directed) relational graphs',
      download_url='...',
      license='MIT',
      install_requires=['numpy',
                        'theano',
                        'pytorch',
                        'rdflib',
                        'scipy',
                        'pandas',
                        'wget'
                        ],
      extras_require={
          'model_saving': ['h5py'],
      },
      package_data={'rgcn_torch': ['README.md', 'data']},
      packages=find_packages())
