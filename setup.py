from setuptools import setup, find_packages

setup(
    name="qutims",
    version="0.2",
    description="QUTIMS is a repository with Quantum Machine Learning algorithms for Multivariate Time Series prediction." \
    "QUTIMS tasks are performed through Quantum Recurrent Neural Networks (QRNNs), which at the NISQ era of quantum" \
    "computing are mainly emulated thorugh classical computer. The repository contains QURECNETS, a 100 % python module to" \
    "train and test Quantum Recurrent Neural Networks and make accurate multivariate time series predictions. QURECNETS" \
    "relies on a specific Density Matrix method that emulates QRNNs. Examples with different datasets are included as a" \
    "guide for a more friendly use.",
    author="J.D. Viqueira",
    author_email="jdviqueira@cesga.es",
    license="GNU General Public License version 3",
    packages=find_packages(),
    install_requires=["numpy"],
)