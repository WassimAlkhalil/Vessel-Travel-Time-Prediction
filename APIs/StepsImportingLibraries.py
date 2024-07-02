"""

PLEASE MAKE SURE THE FOLDER, PATHS AND FILES NAMES ARE CORRECT BEFORE EXECUTING THE API!!

README for Importing Libraries from 'libs' Folder into 'APIs' Folder

This README explains how to import libraries from the 'libs' folder into an API script located in the 'APIs' folder. Follow the steps below:

Folder Structure:
------------------
project/
├── APIs/
│   └── my_api.py
└── libs/
    ├── __init__.py
    └── my_lib.py

Steps to Import Libraries:
--------------------------

1. Using `sys.path.append`:
   -------------------------
   In your API script, modify the `sys.path` to include the path to the 'libs' folder.

   import sys
   import os

   # Add the libs directory to the sys.path
   libs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../libs'))
   sys.path.append(libs_path)

   # Now you can import your libraries
   import my_lib

   def main():
       print(my_lib.hello())

   if __name__ == "__main__":
       main()

2. Using `PYTHONPATH` Environment Variable:
   ----------------------------------------
   Set the `PYTHONPATH` environment variable to include the 'libs' directory.

   On Unix or macOS:
   export PYTHONPATH=$PYTHONPATH:/path/to/libs

   On Windows:
   set PYTHONPATH=%PYTHONPATH%;C:\path\to\libs

   After setting the `PYTHONPATH`, run your API script normally.

3. Creating a Package:
   -------------------
   Ensure the 'libs' folder contains an `__init__.py` file. This file can be empty or contain initialization code for the package.

   Folder structure:
   project/
   ├── APIs/
   │   └── my_api.py
   └── libs/
       ├── __init__.py
       └── my_lib.py

   In your API script (my_api.py), you can then import the library directly:

   from libs import my_lib

   def main():
       print(my_lib.hello())

   if __name__ == "__main__":
       main()

Example:
--------

1. Create the following folder structure:

   project/
   ├── APIs/
   │   └── my_api.py
   └── libs/
       ├── __init__.py
       └── my_lib.py

2. libs/my_lib.py:
   def hello():
       return "Hello from my_lib!"

3. APIs/my_api.py:
   import sys
   import os

   # Add the libs directory to the sys.path
   libs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../libs'))
   sys.path.append(libs_path)

   # Import the library
   import my_lib

   def main():
       print(my_lib.hello())

   if __name__ == "__main__":
       main()

With this setup, running my_api.py will correctly import and use the function from my_lib.py.
"""
