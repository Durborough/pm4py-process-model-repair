from setuptools import setup, find_packages

setup(
    name='pm4py_model_repair',
    version='0.1.7',
    author='Ben Lakhoune on behalf of Durborough',
    author_email='a.b.lakhoune@gmail.com',
    description=' This Algorithm implements a version of a process model repair Algorithm inspired by the paper "Repairing process models to reflect reality" by Dirk Fahland and Wil van der Aalst. DOI: 10.1016/j.is.2013.12.007. This implementation was created by Durborough on GitHub. I only added the setup.py file to make it installable via pip.',
    packages=find_packages(),
    install_requires=[
        'pm4py',
    ],
    py_modules=['process_model_repair_algorithm']
    ,long_description=open('README.md').read(),
)