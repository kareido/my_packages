from setuptools import setup


setup(
    name='my_packages',
    version='0.1',
    packages=['euler_distributed'],
    include_package_data=True,
    install_requires=['torch>=1.0'],
    entry_points={
        'console_scripts': ['lrun = euler_distributed.lrun:lrun'],
    },

    author='Zhe Huang',
    author_email='zhuang334@wisc.edu',
    description='PyTorch Distributed Module for Euler Cluster',
    keywords='euler, distributed, pytorch',
    url='https://github.com/kareido/my_packages',
)

try:
    import euler_distributed as edist
    print('>>> euler_distributed installed <<<')
except Exception as exc:
    print(exc)

