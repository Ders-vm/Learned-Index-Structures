# This makes src/indexes/cpp a Python package
import os
import sys

# Add MinGW DLL directory on Windows
if sys.platform == 'win32':
    mingw_bin = r'C:\mingw64\bin'
    if os.path.exists(mingw_bin):
        os.add_dll_directory(mingw_bin)

from . import btree_cpp

__all__ = ['btree_cpp']