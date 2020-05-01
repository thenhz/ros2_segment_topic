from setuptools import setup

package_name = 'ros2_segment_topic'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='TheNHz',
    maintainer_email='thenhz@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ros2_segment_topic = ros2_segment_topic.segment_and_publish:main',
        ],
    },
)
