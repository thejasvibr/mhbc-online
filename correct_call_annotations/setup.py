from setuptools import setup, find_packages

setup(name='correct_call_annotations',
    version='0.0.0',
    description='extract, verify and correct call annotations',
    author='Thejasvi Beleyur',
    package=find_packages(),
    install_requires=['soundfile','pandas','numpy>1.15']


)
