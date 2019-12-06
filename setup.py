from setuptools import setup


setup(
    name='euler_distributed',
    version='0.1',
    packages=['euler_distributed'],
    include_package_data=True,
    install_requires=['torch>=1.0'],

    author='Zhe Huang',
    author_email='zhuang334@wisc.edu',
    description='PyTorch Distributed Module for Euler Cluster',
    keywords='euler, distributed, pytorch',
    url='https://github.com/kareido/euler_distributed',
)

try:
    import euler_distributed as edist
    print('>>> package installed <<<')
except Exception as exc:
    print(exc)

