Malleability of Locus of Control by Traumatic Events
===================================================

Getting started
---------------

All you need to build the project is an Anaconda Distribution with Python 3.6.
Some packages might show up as not installed. Add them with ``pip install
name`` and rerun the building process until everything works.

To build the project type the next commands in the following order:

        python waf.py configure
        python waf.py build
        python waf.py install

The first command will fail if any one of the required programs cannot be
found.

If the second step fails, try the following in order to localise the problem
(otherwise you may have many parallel processes started and it will be
difficult to find out which one failed):

    python waf.py build -j1

If everything worked without error, you may now find more information on how to
use the project template in "project_documentation/index.html".

If you want to delete all files in the ``build/`` folder, run

  python waf.py distclean


[Todo](TODO.md)
---------------

A list of what has to be done.
