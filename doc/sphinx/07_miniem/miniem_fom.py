#!/usr/bin/env python3

"""
This is a self-contained script that extracts the MiniEM FOM for ATS-5.

This self-contained script extracts the figure of merit (FOM) from MiniEM (ca.
2023) output files. The FOM is the steps-per-second from the BelosLOWS timer
block.
Author: Anthony M. Agelastos <amagela@sandia.gov>
"""


# import Python functions
import sys
import argparse
import os
import logging
import xml.etree.ElementTree as ET
from collections import OrderedDict
import csv


assert sys.version_info >= (3, 5), "Please use Python version 3.5 or later."


# define GLOBAL vars
VERSION = "2.71"
TIMEOUT = 30
IS_ALL = True
EXIT_CODES = {"success": 0, "no file": 1, "bad loop time block": 2}
NUM_DOMAINS = 8


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
            default="output.log",
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


class MiniemFom(object):
    """This class encapsulates the MiniEM FOM."""

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
        self.metrics_cache = OrderedDict()

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

    def _run_fom(self):
        """Extract the FOM."""
        self.logger.debug("Extracting the FOM...")
        # SM_Matrix   # active processes: 56/56
        # |   |   |   |   Stratimikos: BelosLOWS: 726.325 - 78.1426% [5804] {min=726.202, max=726.605, std dev=0.0829577} <       8, 7, 15, 12, 5, 3, 4, 1, 0, 1>
        # MAX MEMORY ALLOCATED: 847166.7 kB

        num_ranks = None
        num_ranks_per_domain = None
        fom = None
        emtime = None
        emiter = None
        emmaxrss = None
        loop_info = []
        is_extract = False
        with open(self.file_name) as fp:
            line = fp.readline()
            while line:
                line = fp.readline()
                if "# active processes" in line and num_ranks is None:
                    l_line = line.split()
                    num_ranks = int(l_line[4].split('/')[0])
                    num_ranks_per_domain = int(round(num_ranks / NUM_DOMAINS))
                if "BelosLOWS" in line and emtime is None and emiter is None:
                    l_line = line.split()
                    emtime = float(l_line[6])
                    emiter = int(l_line[9].strip("[").strip("]"))
                    fom = float(emiter) / emtime
                if "MAX MEMORY ALLOCATED" in line:
                    l_line = line.split()
                    tmp = float(l_line[3])
                    if emmaxrss is None:
                        emmaxrss = tmp
                    if tmp > emmaxrss:
                        emmaxrss = tmp
                    
        if num_ranks is not None and emmaxrss is not None:
            emmaxrss = (emmaxrss * num_ranks) / 1024.0 / 1024.0
        self.metrics_cache['FOM'] = fom
        self.metrics_cache['NumRanks'] = num_ranks
        self.metrics_cache['RanksPerDomain'] = num_ranks_per_domain
        self.metrics_cache['Steps'] = emiter
        self.metrics_cache['Time (sec)'] = emtime
        self.metrics_cache['MaxRSS (GiB)'] = emmaxrss
        self.logger.info("FOM = {}".format(fom))
        self.logger.info("No. Ranks = {}".format(num_ranks))
        self.logger.info("Ranks per Domain = {}".format(num_ranks_per_domain))
        self.logger.info("Steps = {}".format(emiter))
        self.logger.info("Time (sec) = {}".format(emtime))
        self.logger.info("MaxRSS (GiB) = {}".format(emmaxrss))

    def _run_try(self):
        """Extract the metadata."""
        # /usr/projects/hpctest/amagela/ats-5/MiniEM/lb/doc/sphinx/07_miniem/runs--ensemble-1/run-20231203_212259_439273/try-00
        self.logger.debug("Extracting the metadata...")
        empath = os.path.abspath(self.file_name)
        emtry = None
        l_empath = empath.split(os.sep)
        for fldr in l_empath:
            if "try-" in fldr:
                emtry = int(fldr.strip("try-"))
                break
        
        self.metrics_cache['Try'] = emtry
        self.logger.info("Try = {}".format(emtry))

    def _run_size(self):
        """Extract the size."""
        self.logger.debug("Extracting the size...")
        file_name_input = "maxwell-large.xml"
        emsize = None
        empath = os.path.abspath(self.file_name)
        empath = os.path.dirname(empath)
        empath = os.path.join(empath, file_name_input)
        
        # <ParameterList>
        #   <ParameterList name="Mesh">
        #     <ParameterList name="Inline Mesh">
        #       <ParameterList name="Mesh Factory Parameter List">
        #         <Parameter name="X Elements" type="int" value="40" />
        try:
            tree = ET.parse(empath)
            root = tree.getroot()
            for c1 in root:
                if c1.attrib['name'] == 'Mesh':
                    for c2 in c1:
                        if c2.attrib['name'] == 'Inline Mesh':
                            for c3 in c2:
                                if c3.attrib['name'] \
                                    == 'Mesh Factory Parameter List':
                                    for c4 in c3:
                                        if c4.attrib['name'] \
                                            == 'X Elements':
                                            emsizex = int(c4.attrib['value'])
                                        if c4.attrib['name'] \
                                            == 'Y Elements':
                                            emsizey = int(c4.attrib['value'])
                                        if c4.attrib['name'] \
                                            == 'Z Elements':
                                            emsizez = int(c4.attrib['value'])
            emsize = int(round(pow(emsizex * emsizey * emsizez, 1/3), 1))
        except:
            self.logger.critical("Cannot parse {}!".format(file_name_input))
        
        self.metrics_cache['Size'] = emsize
        self.logger.info("Size = {}".format(emsize))

    def _run_csv(self):
        """Put stuff into CSV file."""
        file_csv = self.file_name.replace('.log', '.csv')
        top_row = list(self.metrics_cache.keys())
        bottom_row = [ self.metrics_cache[i] for i in self.metrics_cache ]
        with open(file_csv, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',',
                                   quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow(top_row)
            csvwriter.writerow(bottom_row)

    def run(self):
        """Extract the FOM."""
        self._run_fom()
        self._run_try()
        self._run_size()
        self.metrics_cache['File'] = os.path.abspath(self.file_name)
        self._run_csv()


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
    miniem_fom = MiniemFom(
        logger=logger,
        file_name=cl_args.file,
        is_all=cl_args.all,
    )

    # do work
    miniem_fom.run()

    # exit gracefully
    sys.exit(EXIT_CODES["success"])
