from setuptools import setup, find_packages

setup(
  name = 'umap_pytorch',
  packages = find_packages(exclude=[]),
  include_package_data = True,
  version = '0.0.04',
  license='MIT',
  description = 'Umap port for pytorch',
  author = 'Elio Pascarelli',
  author_email = 'elio@pascarelli.com',
  url = 'https://github.com/elyxlz/umap_pytorch',
  long_description_content_type = 'text/markdown',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'dimensionality reduction',
    'UMAP',
  ],
  install_requires=[
    'einops>=0.3',
    'pynndescent',
    'llvmlite>=0.34.0',
    'torch>=1.6',
    'sklearn',
    'umap-learn',
    'pytorch_lightning',
    'dill',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.9',
  ],
)