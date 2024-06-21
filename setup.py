from setuptools import setup, find_packages

requirements = [
    'numpy==1.26.4',
    'scipy',
    'torch',
    'torchvision',
    'torchmetrics',
    'torch_geometric',
    'gpytorch',
    'pandas',
    'pyarrow',
    'scikit-learn',
    'jupyterlab',
    'matplotlib',
    'seaborn'
]

setup(
    name='gpinfuser',
    version='0.0.1',
    author='Alan Matias',
    author_email='matiasalsm@gmail.com',
    description='Infusing GPs with Deep Kernels and Amortized Inference',
    install_requires=requirements,
    package_dir={'': 'src'},
    packages=find_packages('src')
)