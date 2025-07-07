from setuptools import setup, find_packages

setup(
    name="fleet-shift-analyzer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        line.strip()
        for line in open('requirements.txt').readlines()
        if line.strip() and not line.startswith('#')
    ],
    entry_points={
        'console_scripts': [
            'fleet-shift-analyzer=fleet_shift_analyzer.main:main',
        ],
    },
    python_requires='>=3.8',
)