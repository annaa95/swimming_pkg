from setuptools import find_packages, setup

package_name = 'swimming_pkg'
submodule_path = package_name + '/swimming_mab_robot' 

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, submodule_path],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='anna',
    maintainer_email='annaastolfi28@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'swimmer = swimming_pkg.swimming_node:main',
        ],
    },
)
