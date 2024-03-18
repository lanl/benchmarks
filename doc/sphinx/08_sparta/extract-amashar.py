#!/usr/bin/env python3

"""
This is a self-contained script that extracts a AMASHAR bundle.

This self-contained Python script extracts a AMASHAR bundle and creates the
appropriate directory hierarchy.
Author: Anthony M. Agelastos <amagela@sandia.gov>
"""


# import Python functions
import sys
import argparse
import os
import logging
from collections import OrderedDict
import csv

assert sys.version_info >= (3, 5), "Please use Python version 3.5 or later."


# define GLOBAL vars
VERSION = "2.71"
IS_ALL = True
EXIT_CODES = {"success": 0, "no file": 1}


# define global functions
def print_exit_codes():
    """Print out exit codes."""
    super_str = "exit codes = {"
    for key, value in EXIT_CODES.items():
        super_str += '"{}": {}, '.format(key, value)
    super_str = super_str[:-2]
    super_str += "}"
    return super_str


def is_file(file_name):
    """Check if the file exists and can be read."""
    return os.access(file_name, os.R_OK)


# define classes
class BuildDocHelp(object):
    """Display help."""

    def __init__(self):
        """Initialize object and create argparse entities."""
        my_epilog = print_exit_codes()
        self.parser = argparse.ArgumentParser(
            description="This Python program will extract the figure of merit (FOM) for SPARTA.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            epilog=my_epilog,
        )

        self.parser.add_argument(
            "-a",
            "--all",
            action="store_true",
            default=IS_ALL,
            help="Generate ALL FOM information",
        )

        self.parser.add_argument(
            "-f",
            "--file",
            type=str,
            default="log.sparta",
            help="file name to read",
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
        """Return argparse-parsed arguments for checking workflow state."""
        return self.args


class Amashar(object):
    """This class encapsulates an AMASHAR bundle."""

    def __init__(self, **kwargs):
        """Initialize object and define initial desired build state."""
        # set parameters from object instantiation
        for key, value in kwargs.items():
            setattr(self, key, value)

        # check for required attributes
        required_attr = [
            "logger",
            "file_name",
            "is_all",
        ]
        needed_attr = [item for item in required_attr if not hasattr(self, item)]
        assert len(needed_attr) == 0, (
            "Please ensure object {} has the following required "
            "attributes: {}!".format(self.__class____name__, required_attr)
        )

        # check attributes
        self._check_attr()

    def _check_attr(self):
        """Check object attributes."""
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

        if not is_file(self.file_name):
            self.logger.critical('Cannot read "{}"'.format(self.file_name))
            sys.exit(EXIT_CODES["no file"])

    def _extract_statistics(self):
        """Loop through file and gather statistics."""

        # open up file and get lists of files and bundle their contents accordingly
        files_split = {}
        path_full = ""
        with open(self.file_name) as fp:
            line = fp.readline()
            while line:
                if len(line) > 6:
                    line_check = '##### '
                    line_front = line[0:len(line_check)]
                    if line_front == line_check:
                        path_full = line[len(line_check):].rstrip()
                        files_split[path_full] = []
                files_split[path_full].append(line)
                line = fp.readline()
            
        # fill up the bundles
        for path_full in files_split:
            path_dirname = os.path.dirname(path_full)
            os.makedirs(path_dirname)
            with open(path_full, 'w') as fp:
                self.logger.info("Writing {}".format(path_full))
                fp.writelines(files_split[path_full])
            

    def run(self):
        """Read in SHAR and begin to unpack the bundle."""
        self._extract_statistics()


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
    amashar = Amashar(
        logger=logger,
        file_name=cl_args.file,
        is_all=cl_args.all,
    )

    # do work
    amashar.run()

    # exit gracefully
    sys.exit(EXIT_CODES["success"])
