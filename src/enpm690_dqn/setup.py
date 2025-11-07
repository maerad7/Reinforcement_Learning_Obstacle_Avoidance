import glob
import os
from setuptools import find_packages, setup

package_name = 'enpm690_dqn'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob.glob(os.path.join('launch', 'turtlebot3_ddrl_stage1.launch.py'))),
        ('share/' + package_name + '/launch', glob.glob(os.path.join('launch', 'turtlebot3_drl_stage2.launch.py'))),
        ('share/' + package_name + '/launch', glob.glob(os.path.join('launch', 'turtlebot3_drl_stage3.launch.py'))),
        ('share/' + package_name + '/launch', glob.glob(os.path.join('launch', 'turtlebot3_drl_stage4.launch.py'))),
        ('share/' + package_name + '/launch', glob.glob(os.path.join('launch', 'turtlebot3_drl_stage5.launch.py'))),
        ('share/' + package_name + '/launch', glob.glob(os.path.join('launch', 'turtlebot3_drl_stage6.launch.py'))),
    ],
    install_requires=['setuptools','launch'],
    zip_safe=True,
    maintainer='suhas',
    maintainer_email='suhas99@umd.edu',
    description='ENPM690 Final Project - DQN',
    license='Apache License, Version 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'dqn_envi = enpm690_dqn.dqn_envi:main',
            'dqn_gazebo = enpm690_dqn.dqn_goals:main',
            'dqn_train = enpm690_dqn.dqn:main_train',
            'dqn_test = enpm690_dqn.dqn:main_test',
        ],
    },
)