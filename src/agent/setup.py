from setuptools import find_packages, setup

package_name = 'agent'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    package_data={
        'agent': ['prompts/*.txt'],
    },
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='aswoo55555@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'remembr_agent = agent.remembr_agent:main',
        ],
    },
)
