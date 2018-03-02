Malleability of Locus of Control by Traumatic Events
====================================================

Getting started
---------------

All you need to build the project is an Anaconda Distribution with Python 3.6.
Some packages might show up as not installed. Add them with ``pip install
name`` and rerun the building process until everything works.

To build the project type the next commands in the following order:

        $ python waf.py configure
        $ python waf.py build
        $ python waf.py install

The first command will fail if any one of the required programs cannot be
found.

If the second step fails, try the following in order to localise the problem
(otherwise you may have many parallel processes started and it will be
difficult to find out which one failed):

        $ python waf.py build -j1

If everything worked without error, you may now find more information on how to
use the project template in "project_documentation/index.html".

If you want to delete all files in the ``build/`` folder, run

        $ python waf.py distclean

To run the test suite, type

    $ tox

or to run specific tests described in ``tox.ini`` under ``envlist`` run

    $ tox -e <env_name>

This will run ``pytest`` for checking the implementation and produce a coverage
report, perform linting on documentation (``doc8``) and code (``flake8``).


Additional resources
--------------------

- [statsmodels tutorial on formulas](http://www.statsmodels.org/dev/example_formulas.html)
- [statsmodels tutorial on standard errors](http://www.vincentgregoire.com/standard-errors-in-python/)
- [PCA tutorial](https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html)
- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)