from setuptools import find_packages, setup

setup(
    name='remembr_mcp',
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    install_requires=[
        'mcp>=1.0.0',
        'qdrant-client>=1.14.0',
        'langchain-huggingface>=0.1.0',
    ],
    entry_points={
        'console_scripts': [
            'remembr_mcp = remembr_mcp.__main__:main',
        ],
    },
)
