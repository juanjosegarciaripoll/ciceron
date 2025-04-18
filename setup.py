from setuptools import find_packages, setup, Extension
import glob
import numpy as np
import sys


def get_cython_options():
    return {
        "include_dirs": [np.get_include()],
    }


# This flag controls whether we build the library with bounds checks and
# other safety measures. Useful when testing where a code breaks down;
# but bad for production performance
debug_library = False
extra_compile_args = []
try:
    from Cython.Build import cythonize
    import Cython.Compiler.Options

    # See https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html
    # for a deeper explanation of the choices here
    # Cython.Compiler.Options.docstring = False
    Cython.Compiler.Options.error_on_uninitialized = True
    if not debug_library:
        directives = {
            "language_level": "3",  # We assume Python 3 code
            "boundscheck": False,  # Do not check array access
            "wraparound": False,  # a[-1] does not work
            "always_allow_keywords": False,  # Faster calling conventions
            "cdivision": True,  # No exception on zero denominator
            "initializedcheck": False,  # We take care of initializing cdef classes and memory views
            "overflowcheck": False,
            "cpp_locals": True,
        }
        if sys.platform == "linux":
            # We assume GCC or other compilers with compatible command line
            extra_compile_args = ["-O3", "-ffast-math"]
        else:
            # We assume Microsoft Visual C/C++ compiler
            extra_compile_args = ["/Ox", "/fp:fast", "/arch:AVX2"]
    else:
        directives = {
            "language_level": "3",  # We assume Python 3 code
            "always_allow_keywords": False,  # Faster calling conventions
            "boundscheck": True,
            "initializedcheck": True,
            "wraparound": False,
        }
    use_cython = True
except ImportError:
    use_cython = False

# All Cython files with unix pathnames
cython_files = [s.replace("\\", "/") for s in glob.glob("**/*.pyx", recursive=True)]
extension_names = [".".join(f[:-4].split("/")[1:]) for f in cython_files]
print(extension_names)
print(cython_files)
extensions = [
    Extension(
        name,
        [file],
        language="c++",
        extra_compile_args=extra_compile_args,
        **get_cython_options(),
    )
    for name, file in zip(extension_names, cython_files)
]
if use_cython:
    extensions = cythonize(extensions, compiler_directives=directives)

setup(ext_modules=extensions)
