from setuptools import setup, find_packages

setup(
    name='my_environment',
    version='0.0.1',
    install_requires=['gym==0.21.0', 'scipy==1.7.3','pygame==2.1.2', 'numpy==1.21.*', 'wandb'],
    include_package_data=True,
    package_dir={"": "my_environment"}
    )
