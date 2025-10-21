# From Jacobus G.M. van der Linden “STreeD”
# https://github.com/AlgTUDelft/pystreed 

# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_include as get_pybind_include
from setuptools import setup
from glob import glob

# Define package metadata
package_name = 'pysortd'
extension_name = 'csortd'
__version__ = "0.0.1"

ext_modules = [
    Pybind11Extension(package_name + '.' + extension_name,
        sorted(glob("src/**/*.cpp", recursive = True)),
        include_dirs = ["include"] + [get_pybind_include()],
        define_macros = [('VERSION_INFO', __version__), 
                         ('WITH_PYBIND', 1)], 
        language='c++',
        cxx_std=17
    )
]

setup(
    name=package_name,
    version=__version__,
    ext_modules=ext_modules,
    dev_requires=['pytest'],
    install_requires=['pandas', 'numpy']
)