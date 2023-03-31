#!/usr/bin/env python
# This is a self-contained script that builds documentation for ATS Benchmarks!
# Author: Anthony M. Agelastos <amagela@sandia.gov>


# import Python functions
import sys

assert sys.version_info >= (3, 5), "Please use Python version 3.5 or later."
import argparse
import os
import logging
import shutil
import textwrap
import subprocess
import glob


# define GLOBAL vars
VERSION = "2.71"
TIMEOUT = 30
IS_ALL = True
IS_PDF = False
IS_HTML = False
IS_MAN = False
IS_MARKDOWN = False
DIR_BUILD = "_build"
EXIT_CODES = {"success": 0, "no app": 1, "app run issue": 2, "directory issue": 3}


# define global functions
def print_exit_codes():
    """This function prints out exit codes."""
    super_str = "exit codes = {"
    for key, value in EXIT_CODES.items():
        super_str += '"{}": {}, '.format(key, value)
    super_str = super_str[:-2]
    super_str += "}"
    return super_str


def make_dir(logger, mdir):
    """This makes a directory."""
    assert isinstance(logger, logging.RootLogger), "Pass appropriate logging object!"

    bdir = os.path.dirname(mdir)

    # if it already exists...
    if os.path.isdir(mdir):
        logger.info('Directory "{}" is already present.'.format(mdir))
        return

    # if base directory is not writable...
    logger.debug('Directory "{}" is not present.'.format(mdir))
    if not os.access(bdir, os.W_OK):
        logger.critical('Base directory "{}" is not writable!'.format(bdir))
        sys.exit(EXIT_CODES["directory issue"])

    # finally make the friggin thing
    try:
        os.makedirs(mdir)
    except OSError:
        logger.critical('Creation of directory "{}" failed!'.format(mdir))
        sys.exit(EXIT_CODES["directory issue"])
    else:
        logger.info('Creation of directory "{}" was successful'.format(mdir))


def run_app(logger, args):
    """This executes application and args defined in args."""

    # subprocess.run(
    #     [],
    #     stdin=None,
    #     input=None,
    #     stdout=None,
    #     stderr=None,
    #     capture_output=False,
    #     shell=False,
    #     cwd=None,
    #     timeout=TIMEOUT,
    #     check=True,
    #     encoding=None,
    #     errors=None,
    #     text=None,
    #     env=None,
    #     universal_newlines=None,
    # )

    assert isinstance(logger, logging.RootLogger), "Pass appropriate logging object!"

    # generate Makefile and other supported files
    try:
        foo = subprocess.run(args, timeout=TIMEOUT)
        if foo.returncode == 0:
            logger.info("Application {} exited cleanly.".format(args))
        else:
            logger.critical(
                "Application {} exited with non-zero ({}) exit code!".format(
                    args, foo.returncode
                )
            )
            sys.exit(EXIT_CODES["app run issue"])
    except:
        logger.critical("Application {} had issues!".format(args))
        sys.exit(EXIT_CODES["app run issue"])


def check_app(logger, name_app):
    """This checks for a valid installation of an application."""

    assert isinstance(logger, logging.RootLogger), "Pass appropriate logging object!"
    is_app = shutil.which(name_app) is not None
    if is_app:
        logger.info("Found installation of {}.".format(name_app))
    else:
        logger.critical("Did not find installation of {}!".format(name_app))
    return is_app


def do_cmd(command):
    """Execute command."""
    return subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        universal_newlines=True,
    )


def do_wrk_scripts(logger):
    """This executes found work scripts."""

    dir_wrk = "wrk"
    if not os.path.isdir(dir_wrk):
        logger.info(
            "Did not find {} directory; not executing any scripts.".format(dir_wrk)
        )
        return

    dir_base = os.getcwd()
    os.chdir("wrk")

    # execute BSH/BASH scripts
    files = glob.glob("*.sh")
    files.extend(glob.glob("*.bsh"))
    files.extend(glob.glob("*.bash"))
    for fl in files:
        logger.info("Executing BSH/BASH script {}...".format(fl))
        do_cmd("bash ./" + fl)

    os.chdir(dir_base)


# define classes
class BuildDocHelp(object):
    """This is a class that encapsulates the command line processing for
    building documentation."""

    def __init__(self):
        """Initialize object and create argparse entities."""

        my_epilog = print_exit_codes()
        self.parser = argparse.ArgumentParser(
            description="This Python program will build the documentation for ADPS.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            epilog=my_epilog,
        )

        self.parser.add_argument(
            "-a",
            "--all",
            action="store_true",
            default=IS_ALL,
            help="Generate ALL export types",
        )

        self.parser.add_argument(
            "-p",
            "--pdf",
            action="store_true",
            default=IS_PDF,
            help="Generate PDF export type",
        )

        self.parser.add_argument(
            "--html",
            action="store_true",
            default=IS_HTML,
            help="Generate HTML export type",
        )

        self.parser.add_argument(
            "--man",
            action="store_true",
            default=IS_MAN,
            help="Generate UNIX manual page",
        )

        self.parser.add_argument(
            "--markdown",
            action="store_true",
            default=IS_MARKDOWN,
            help="Generate Markdown",
        )

        self.parser.add_argument(
            "-l",
            "--logLevel",
            type=str,
            default="info",
            choices=("info", "debug", "warning"),
            help="logging level",
        )

        self.parser.add_argument(
            "-v", "--version", action="version", version="%(prog)s {}".format(VERSION)
        )

        self.args = self.parser.parse_args()

    def get_args(self):
        """This returns argparse-parsed arguments for checking workflow
        state."""
        return self.args


class BuildDoc(object):
    """This class encapsulates the build of ADPS documentation."""

    def __init__(self, **kwargs):
        """Initialize object and define initial desired build state."""

        # set parameters from object instantiation
        for key, value in kwargs.items():
            setattr(self, key, value)

        # check for required attributes
        required_attr = [
            "logger",
            "is_all",
            "is_pdf",
            "is_html",
            "is_man",
            "is_markdown",
        ]
        needed_attr = [item for item in required_attr if not hasattr(self, item)]
        assert len(needed_attr) == 0, (
            "Please ensure object {} has the following required "
            "attributes: {}!".format(self.__class____name__, required_attr)
        )

        # check attributes
        self._check_attr()

    def _check_attr(self):
        """This checks object attributes."""

        # check inputs
        assert isinstance(
            self.logger, logging.RootLogger
        ), "Pass appropriate logging object to {}!".format(self.__class__.__name__)
        if not isinstance(self.is_all, bool):
            tmp = bool(self.is_all)
            self.logger.critical(
                "Type issue with is_all within {} (should be bool, is {}); converted to bool and is now {}.".format(
                    self.__class__.__name__, type(self.is_all), tmp
                )
            )
            self.is_all = tmp
        if not isinstance(self.is_pdf, bool):
            tmp = bool(self.is_pdf)
            self.logger.critical(
                "Type issue with is_pdf within {} (should be bool, is {}); converted to bool and is now {}.".format(
                    self.__class__.__name__, type(self.is_pdf), tmp
                )
            )
            self.is_pdf = tmp
        if not isinstance(self.is_html, bool):
            tmp = bool(self.is_html)
            self.logger.critical(
                "Type issue with is_html within {} (should be bool, is {}); converted to bool and is now {}.".format(
                    self.__class__.__name__, type(self.is_html), tmp
                )
            )
            self.is_html = tmp
        if not isinstance(self.is_man, bool):
            tmp = bool(self.is_man)
            self.logger.critical(
                "Type issue with is_man within {} (should be bool, is {}); converted to bool and is now {}.".format(
                    self.__class__.__name__, type(self.is_man), tmp
                )
            )
            self.is_man = tmp
        if not isinstance(self.is_markdown, bool):
            tmp = bool(self.is_markdown)
            self.logger.critical(
                "Type issue with is_markdown within {} (should be bool, is {}); converted to bool and is now {}.".format(
                    self.__class__.__name__, type(self.is_markdown), tmp
                )
            )
            self.is_markdown = tmp

        # make inputs consistent
        if self.is_pdf:
            self.is_all = False
        if self.is_html:
            self.is_all = False
        if self.is_man:
            self.is_all = False
        if self.is_markdown:
            self.is_all = False
        elif self.is_all:
            self.is_pdf = True
            self.is_html = True
            self.is_man = True
            self.is_markdown = True

        # check if applications are installed
        self._check_apps()

        # check if build, etc. directories are ready
        self._check_dirs()

    def _check_apps(self):
        """This checks for valid installations of needed software."""
        is_app = []
        is_app.extend([check_app(self.logger, "sphinx-build")])
        # is_app.extend([check_app(self.logger, "pdflatex")])
        is_app.extend([check_app(self.logger, "make")])
        if sum(is_app) != len(is_app):
            sys.exit(EXIT_CODES["no app"])

    def _check_dirs(self):
        """This checks if needed directories are present."""
        path_absdir_thisscript = os.path.dirname(os.path.abspath(__file__))
        path_absdir_build = os.path.join(path_absdir_thisscript, DIR_BUILD)
        self.logger.debug('Build directory is "{}".'.format(path_absdir_build))
        make_dir(self.logger, path_absdir_build)

    def _build_pdf(self):
        """This builds the documentation with exporting to PDF."""

        if not self.is_pdf:
            return
        self.logger.info("Building PDF...")

        run_app(self.logger, ["sphinx-build", "-b", "latex", ".", "_build"])
        run_app(self.logger, ["make", "latexpdf"])

    def _build_html(self):
        """This builds the documentation with exporting to HTML."""

        if not self.is_html:
            return
        self.logger.info("Building HTML...")

        run_app(self.logger, ["sphinx-build", "-b", "html", ".", "_build"])
        run_app(self.logger, ["make", "html"])

    def _build_man(self):
        """This builds the documentation with exporting to UNIX manual format."""

        if not self.is_man:
            return
        self.logger.info("Building UNIX manual...")

        run_app(self.logger, ["sphinx-build", "-b", "man", ".", "_build"])
        run_app(self.logger, ["make", "man"])

    def _build_markdown(self):
        """This builds the documentation with exporting to Markdown format."""

        if not self.is_markdown:
            return
        self.logger.info("Building Markdown...")

        run_app(self.logger, ["sphinx-build", "-b", "markdown", ".", "_build"])
        run_app(self.logger, ["make", "markdown"])

    def build_doc(self):
        """This builds the documentation."""
        self.logger.info("Building documentation...")
        self._build_pdf()
        self._build_html()
        self._build_man()
        self._build_markdown()


# do work
if __name__ == "__main__":

    # manage command line arguments
    build_doc_help = BuildDocHelp()
    cl_args = build_doc_help.get_args()

    # manage logging
    int_logging_level = getattr(logging, cl_args.logLevel.upper(), None)
    if not isinstance(int_logging_level, int):
        raise ValueError("Invalid log level: {}!".format(cl_args.logLevel))
    logging.basicConfig(
        format="%(levelname)s - %(asctime)s - %(message)s", level=int_logging_level
    )
    logging.debug("Set logging level to {}.".format(cl_args.logLevel))
    logger = logging.getLogger()

    # manage worker object
    build_doc = BuildDoc(
        logger=logger,
        is_all=cl_args.all,
        is_pdf=cl_args.pdf,
        is_html=cl_args.html,
        is_man=cl_args.man,
        is_markdown=cl_args.markdown,
    )

    # do work
    do_wrk_scripts(logger)
    build_doc.build_doc()

    # exit gracefully
    sys.exit(EXIT_CODES["success"])
