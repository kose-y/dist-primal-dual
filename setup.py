from setuptools import setup

setup(name="dist_pd", 
    version="0.1",
    description="Distributed primal-dual algorithms on TensorFlow for multiple devices",
    author="Seyoon Ko",
    author_email="syko0507@snu.ac.kr",
    url="https://github.com/kose-y/dist_pd",
    packages=['dist_pd'],
    install_requires=['numpy>=1.13','scipy>=1.0.0', 'h5py>=2.8.0'],
    extras_require={
        'tensorflow': ['tensorflow>=1.2.0'],
        'tensorflow with gpu': ['tensorflow-gpu>=1.5l0']
    },
    keywords=['monotone operator theory', 'non-smooth optimization', 'operator splitting', 'sparsity', 'distributed computing', 'multi-GPU'],
    license="MIT"
)

