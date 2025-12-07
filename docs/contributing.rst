Contributing to DataTools
=========================

We welcome contributions to DataTools! This document provides guidelines for contributing to the project.

Getting Started
---------------

1. Fork the repository on GitHub
2. Clone your fork locally::

    git clone https://github.com/your-username/DataTools.git
    cd DataTools

3. Create a virtual environment::

    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

4. Install development dependencies::

    pip install -e .
    pip install -r requirements.txt

Development Workflow
--------------------

1. Create a new branch for your feature or bugfix::

    git checkout -b feature-name

2. Make your changes

3. Test your changes::

    python -m pytest tests/

4. Commit your changes::

    git add .
    git commit -m "Description of changes"

5. Push to your fork::

    git push origin feature-name

6. Create a Pull Request on GitHub

Code Style
----------

* Follow PEP 8 style guidelines
* Use meaningful variable and function names
* Add docstrings to all functions and classes
* Keep functions focused and modular

Documentation Style
~~~~~~~~~~~~~~~~~~~

We use Google-style docstrings. Example::

    def function_name(param1, param2):
        """Brief description of function.

        Longer description if needed.

        Args:
            param1 (type): Description of param1
            param2 (type): Description of param2

        Returns:
            type: Description of return value

        Raises:
            ValueError: Description of when this is raised
        """
        pass

Testing
-------

All new features should include tests. We use pytest for testing.

Running Tests
~~~~~~~~~~~~~

Run all tests::

    pytest

Run specific test file::

    pytest tests/test_tiff2zarr.py

Run with coverage::

    pytest --cov=DataTools

Writing Tests
~~~~~~~~~~~~~

Example test structure::

    import pytest
    from DataTools.DataFormats import tiff2zarr

    def test_tiff_conversion():
        """Test TIFF to Zarr conversion."""
        input_file = 'test_data/input.tiff'
        output_file = 'test_data/output.zarr'

        result = tiff2zarr.convert(input_file, output_file)

        assert result is not None
        # Add more assertions

Documentation
-------------

Building Documentation
~~~~~~~~~~~~~~~~~~~~~~

Documentation is built using Sphinx. To build locally::

    cd docs
    make html

The built documentation will be in ``docs/_build/html/``.

Adding Documentation
~~~~~~~~~~~~~~~~~~~~

* Add docstrings to all new functions and classes
* Update relevant .rst files in the docs/ directory
* Add examples for new features
* Update the changelog

Submitting Changes
------------------

Pull Request Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~

* Include a clear description of the changes
* Reference any related issues
* Ensure all tests pass
* Update documentation as needed
* Add your name to the contributors list

Code Review Process
~~~~~~~~~~~~~~~~~~~

1. Maintainers will review your pull request
2. Address any requested changes
3. Once approved, your changes will be merged

Reporting Bugs
--------------

Bug reports should include:

1. Description of the bug
2. Steps to reproduce
3. Expected behavior
4. Actual behavior
5. Environment details (OS, Python version, DataTools version)

Submit bug reports as GitHub issues.

Feature Requests
----------------

We welcome feature requests! Please:

1. Check if the feature already exists or has been requested
2. Describe the feature and its use case
3. Explain why it would be valuable
4. Provide examples if possible

Submit feature requests as GitHub issues with the "enhancement" label.

Code of Conduct
---------------

* Be respectful and inclusive
* Welcome newcomers
* Focus on constructive feedback
* Respect differing opinions

Development Tips
----------------

Setting Up IDE
~~~~~~~~~~~~~~

For VS Code, recommended extensions:

* Python
* Pylance
* Python Docstring Generator

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

Some tests may require environment variables::

    export DATATOOLS_TEST_DATA=/path/to/test/data

Debugging
~~~~~~~~~

Use Python's built-in debugger::

    import pdb; pdb.set_trace()

Or use IDE debugging features.

Release Process
---------------

(For maintainers)

1. Update version in VERSION file
2. Update CHANGELOG.rst
3. Create release commit
4. Tag release::

    git tag -a v0.0.2 -m "Release version 0.0.2"
    git push origin v0.0.2

5. Build and upload to PyPI::

    python -m build
    python -m twine upload dist/*

Questions?
----------

If you have questions about contributing, please:

* Check existing documentation
* Search closed issues
* Open a new issue with the "question" label

Thank you for contributing to DataTools!
