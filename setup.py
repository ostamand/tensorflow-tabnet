from setuptools import setup, find_packages

REQUIREMENTS = [
    "tensorflow>=2.0.0",
    "tensorflow-addons>=0.11.1",
    "pandas>=1.1.0"
]

setup(
    name="tabnet",
    version="0.1",
    author="Olivier St-Amand",
    author_email="olivier.st.amand.1@gmail.com",
    description="TensorFlow 2 implementation of TabNet",
    license="MIT",
    packages=find_packages(exclude=["tests"]),
    python_requires=">=3.5.0",
    install_requires=REQUIREMENTS,
)
