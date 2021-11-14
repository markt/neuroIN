from setuptools import setup, find_packages

setup(
   name='neuroIN',
   version='0.1.0',
   author='Mark Taylor',
   author_email='mark.a.taylor.gr@dartmouth.edu',
   packages=find_packages(where="src"),
   scripts=[],
   url='https://github.com/markt/neuroin',
   license='LICENSE',
   description='neuroIN is a deep learning package for neuroimaging data',
   long_description=open('README.md').read(),
   install_requires=[
       "pytest",
   ],
)
