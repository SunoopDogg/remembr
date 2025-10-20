from setuptools import find_packages, setup

package_name = 'vila'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['captioner']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'captioner = vila.captioner:main',
        ],
    },
)
