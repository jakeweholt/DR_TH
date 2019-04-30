from setuptools import setup

setup(
    name='loan_default_predictor',
    packages=['model'],
    include_package_data=True,
    install_requires=[
        'flask',
    ],
)