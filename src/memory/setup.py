from setuptools import find_packages, setup

package_name = 'memory'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='airo-workstation',
    maintainer_email='aswoo55555@gmail.com',
    description='ReMEmbR memory building node (VLM captioner + pose)',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'captioner = memory.captioner:main',
            'captioner_once = memory.captioner_once:main',
        ],
    },
)
