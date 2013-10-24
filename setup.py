try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

try:
   import pypandoc
   description = pypandoc.convert('README.md', 'rst')
except (IOError, ImportError):
   description = 'Basic implementation of Nelder-Mead Simplex algorithm.'

setup(
    name='Simplex',
    version='0.1.0',
    author='Jakub Konka',
    author_email='kubkon@gmail.com',
    packages=['simplex', 'simplex.test'],
    url='https://github.com/kubkon/simplex',
    license='LICENSE.txt',
    description='Basic implementation of Nelder-Mead Simplex algorithm.',
    long_description=description,
    install_requires=[
        "numpy>=1.7.1",
        "matplotlib>=1.3.0",
    ],
)