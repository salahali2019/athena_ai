from setuptools import setup, find_packages

setup(
    name='athena_ai',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch>=1.7.0',
        'numpy>=1.19.5',
        'matplotlib>=3.3.4',
        'scikit-learn>=0.24.1',
        'Pillow>=8.1.0',
    ],
    author='Salah Ali',
    author_email='salah.ali2019@gmail.com',
    description='A collection of PyTorch models for image classification',
    keywords='pytorch machine-learning deep-learning image-classification',
    url='https://github.com/salahali2019/athena',
    classifiers=[
        'Development Status :: testing',
        'Intended Audience :: Developers',
        'Programming Language :: Python ',
    ],
    test_suite='tests',
    tests_require=[
        'unittest',
    ],
)
