.. _DocProject:

*********************
Project Documentation
*********************

This appendix chapter will discuss this project's documentation including its
dependencies (see :ref:`DocDependencies`), building it (see :ref:`DocBuilding`),
and contributing (see :ref:`DocContributing`) to it.


.. _DocDependencies:

Dependencies
============

The documentation is written using the Sphinx Python Documentation Generator
[Sphinx]_. Sphinx is built atop Python [Python]_ and uses reStructuredText
[RST]_ as its lightweight markup language. Sphinx can output to multiple
formats. This documentation targets HTML, PDF (via LaTeX [LaTeX]_), Unix manual
pages, and Markdown. So, installations of Python (e.g., Miniconda [Miniconda]_)
and LaTeX (e.g., TeX Live [TeXLive]_) are needed as Sphinx dependencies.
Additionally, the theme being used for HTML rendering (i.e., "sphinx_rtd_theme"
[RTDTheme]_) is developed by Read the Docs, Inc. [ReadTheDocs]_; this, too, is a
dependency. Outputting to Markdown requires the "sphinx-markdown-builder"
package [MarkdownBuilder]_. The steps for installation are provided below.

1. Install a Python distribution. This project recommends Miniconda which will
   work on Microsoft Windows, Apple macOS, and GNU/Linux distributions. Refer to
   their site [Miniconda]_ for more information.

.. note::

   The Miniconda installer will automatically add items to your shell
   initialization file (e.g., ``~/.bashrc`` for Bash) to load Miniconda into the
   environment. If you do not want Miniconda in your environment all of the
   time, then you can add a function to do this when invoked; an example for
   Bash is below::

       envconda()
       {
           tmppath="/path/to/your/miniconda-3/bin"
           eval "$(${tmppath}/conda shell.bash hook)"
           [[ ":$PATH:" != *":${tmppath}:"* ]] && export PATH="${tmppath}:${PATH}"
       }
       export -f envconda

.. note::

   An easy way to deal with typical proxy issues with the ``conda`` command is
   to create your Conda RC file (e.g., ``~/.condarc`` on Apple macOS and
   GNU/Linux distributions). The contents of this file for me, as an example, is
   below::

       ssl_verify: false
       proxy_servers:
         http: http://nouser:nopass@proxy.sandia.gov:80
         https: http://nouser:nopass@proxy.sandia.gov:80

2. It is recommended to create a Miniconda environment specifically for building
   this documentation. This can be done with the command(s) below (which creates
   an environment named ``docs`` and switches into it)::

       conda create --name docs
       conda activate docs

3. Install Sphinx within your Python distribution. If you are using Miniconda,
   then this can be performed with the following command within the
   aforementioned ``docs`` environment::

       conda install sphinx

4. Install the "sphinx-rtd-theme" theme with the following command::

       pip install --upgrade \
           --trusted-host pypi.org --trusted-host files.pythonhosted.org \
           --proxy proxy.sandia.gov:80 sphinx-rtd-theme

.. note::

   Miniconda has a version of the "sphinx_rtd_theme" package, however it is not
   updated at a desired frequency.

5. Install TeX Live [TeXLive]_ for your system (e.g., MacTeX [MacTeX]_ for Apple
   macOS) with one of the appropriate URLs provided.

6. Install the "sphinx-markdown-builder" theme with the following command::

       pip install --upgrade \
           --trusted-host pypi.org --trusted-host files.pythonhosted.org \
           --proxy proxy.sandia.gov:80 sphinx-markdown-builder


.. _DocBuilding:

Building
========

Building the documentation is mostly managed through the ``build_doc.py`` Python
script. Its help page can be viewed with the command ``build_doc.py -h``.
Running the command sans command line parameters will generate the HTML and PDF
builds of the documentation. This script puts the build files and final outputs
within the ``_build`` directory.

.. note::

   If the ``_build`` directory is already present and you are about to do a new
   build, you may want to delete it beforehand. 

The command to automatically remove ``_build`` if it already exists, build the
documentation, and to pipe the output to a log file is below.

.. code-block:: bash

   rm -rf _build ; ./build_doc.py --all 2>&1 | tee output.log

If successful, the following top-level files are made. They can be opened with
common tools, e.g., Mozilla Firefox for the HTML and Adobe Acrobat for the PDF.

#. **HTML**: ``_build/html/index.html``
#. **Markdown**: ``_build/markdown/index.md``
#. **PDF**: ``_build/latex/tempi.pdf``
#. **Unix manpage**: ``_build/man/tempi.1``

If things are not successful, then peruse the output from the build (e.g.,
captured within ``output.log`` from above). If you are debugging something, then
it may be desired to only build the HTML and not the other targets (e.g., PDF)
since the HTML builds far faster; consult the ``build_doc.py`` help page for
information on how to achieve this.


.. _DocContributing:

Contributing
============

This project welcomes everyone who has a desire to contribute. Please feel free
to provide us with feedback. If you wish to modify the documentation, then there
are some notes below that may be helpful.

* Top-level Sphinx Documentation is in the `Sphinx master top-level contents
  <https://www.sphinx-doc.org/en/master/contents.html>`_.

  * A resource for reStructuredText (RST) is in the `Sphinx RST top-level document 
    <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_.

* Documentation should look good in all formats prior to a commit being pushed
  into branch ``master``.
* All citations should be provided in an appropriate IEEE format.
* This project recommends `VS Code <https://code.visualstudio.com>`_, `Atom
  <https://atom.io>`_, `GNU Emacs <https://www.gnu.org/software/emacs/>`_, and
  `Vim <https://www.vim.org>`_/`Neovim <https://neovim.io>`_ editors for
  development. All of these are cross-platform and support Microsoft Windows,
  Apple macOS, and GNU/Linux distributions. The Atom editor also, apparently,
  has packages that support preview rendering of RST files.



Build Script
============

The aforementioned script that builds this is replicated below for reference.

.. literalinclude:: ../build_doc.py
   :language: python


.. [Sphinx] G. Brandl, 'Overview -- Sphinx 4.0.0+ documentation', 2021. [Online]. Available: https://www.sphinx-doc.org. [Accessed: 12- Jan- 2021]
.. [Python] Python Software Foundation, 'Welcome to Python.org', 2021. [Online]. Available: https://www.python.org. [Accessed: 12- Jan- 2021]
.. [RST] Docutils Authors, 'A ReStructuredText Primer -- Docutils 3.0 documentation', 2015. [Online]. Available: https://docutils.readthedocs.io/en/sphinx-docs/user/rst/quickstart.html. [Accessed: 12- Jan- 2021]
.. [LaTeX] The LaTeX Project, 'LaTeX - A document preparation system', 2021. [Online]. Available: https://www.latex-project.org. [Accessed: 12- Jan- 2021]
.. [Miniconda] Anaconda, Inc., 'Miniconda -- Conda documentation', 2017. [Online]. Available: https://docs.conda.io/en/latest/miniconda.html. [Accessed: 12- Jan- 2021]
.. [TeXLive] TeX Users Group, 'TeX Live - TeX Users Group', 2020. [Online]. Available: https://www.tug.org/texlive/. [Accessed: 12- Jan- 2021]
.. [RTDTheme] Read the Docs, Inc., 'GitHub - readthedocs/sphinx_rtd_theme: Sphinx theme for readthedocs.org', 2021. [Online]. Available: https://github.com/readthedocs/sphinx_rtd_theme. [Accessed: 12- Jan- 2021]
.. [ReadTheDocs] Read the Docs, Inc., 'Home | Read the Docs', 2021. [Online]. Available: https://readthedocs.org. [Accessed: 12- Jan- 2021]
.. [MacTeX] MacTeX Developers, 'MacTeX - TeX Users Group', TuG Users Group, 2020. [Online]. Available: https://www.tug.org/mactex. [Accessed: 12- Jan- 2021]
.. [MarkdownBuilder] J. Risser and M. Brett, 'GitHub - clayrisser/sphinx-markdown-builder: sphinx builder that outputs markdown files.', 2022. [Online]. Available: https://github.com/clayrisser/sphinx-markdown-builder. [Accessed: 8- Feb- 2022]
