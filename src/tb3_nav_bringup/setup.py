import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'tb3_nav_bringup'


def get_recursive_data_files(base_dir):
    data_files = []

    for root, dirs, files in os.walk(base_dir):
        if files:
            install_path = 'share/' + package_name + '/' + root
            file_paths = [os.path.join(root, f) for f in files]
            data_files.append((install_path, file_paths))

    return data_files


setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/maps', glob('maps/*')),
        ('share/' + package_name + '/params', glob('params/*')),
        ('share/' + package_name + '/rviz', glob('rviz/*')),
        ('share/' + package_name + '/urdf', glob('urdf/*')),
        ('share/' + package_name + '/worlds', glob('worlds/*.world')),
    ] + get_recursive_data_files('worlds/models') + get_recursive_data_files('worlds/photos'),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'query_navigator = tb3_nav_bringup.query_navigator:main',
        ],
    },
)
