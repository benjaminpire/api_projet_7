from setuptools import setup, find_packages

setup(
    name='Projet_7',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # List your project dependencies here
        'numpy',
        'requests',
        # Add other dependencies as needed
    ],
    entry_points={
        'console_scripts': [
            'python-app=your_package.module:main_function',
        ],
    },
)