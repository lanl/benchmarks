#!/usr/bin/env python3

"""
This is a self-contained script that extracts the SPARTA FOM for ATS-5.

This self-contained script extracts the figure of merit (FOM) from SPARTA (ca.
early 2023) log.sparta output files. The FOM is the harmonic mean of the
computed Mega-cell-steps-per-second from the Loop timer block between 5 and 10
minutes of wall time.
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
TIMEOUT = 30
IS_ALL = True
NUM_RANKS_PER_NODE = 112
EXIT_CODES = {"success": 0, "no file": 1, "bad loop time block": 2}


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
            "-n",
            "--numRanksPerNode",
            type=int,
            default=NUM_RANKS_PER_NODE,
            help="number of MPI rank per node",
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


class SpartaFom(object):
    """This class encapsulates the build of ADPS documentation."""

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
            "num_ranks_per_node",
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

    def _check_start(self, line):
        """Check if this is the start of the Loop time block."""
        return "Step CPU Np Natt Ncoll Maxlevel" in line

    def _check_end(self, line):
        """Check if this is the end of the Loop time block."""
        return "Loop time of " in line and "steps with" in line

    def _extract_line(self, line):
        """Extract and parse the line."""
        l_line = line.split()
        if len(l_line) != 6:
            self.logger.critical("Loop time block not sized appropriately!")
            sys.exit(EXIT_CODES["bad loop time block"])
        n_line = []
        n_line.append(int(l_line[0]))
        n_line.append(float(l_line[1]))
        n_line.append(int(l_line[2]))
        n_line.append(int(l_line[3]))
        n_line.append(int(l_line[4]))
        n_line.append(int(l_line[5]))
        return n_line

    def _compute_fom(self, block):
        """Compute the FOM."""
        vals = []
        start = 300.0
        finish = 600.0
        for line in block:
            if line[1] >= finish:
                break
            if line[1] > start:
                fom = line[2] * line[0] / line[1] / 1000000
                fom = 1 / fom
                vals.append(fom)
        num_vals = len(vals)

        hmean_fom = 0
        hmean_denom = 0
        if num_vals != 0:
            for item in vals:
                hmean_denom = hmean_denom + item
            hmean_fom = num_vals / hmean_denom

        return hmean_fom

    def _check_ranks(self, line):
        """Extract the number of MPI ranks."""
        # Running on 7168 MPI task(s)
        return "Running on" in line and "MPI task(s)" in line

    def _check_ppc(self, line):
        """Extract the PPC."""
        # variable ppc equal 35
        l_line = line.split()
        bool_1 = "variable" in line and "ppc" in line and "equal" in line
        bool_2 = len(l_line) == 4
        bool_3 = True
        if bool_2:
            try:
                m_l = float(l_line[3])
                if m_l > 1:
                    bool_3 = True
            except:
                bool_3 = False
        return bool_1 and bool_2 and bool_3

    def _check_l(self, line):
        """Extract L."""
        # variable L equal 8
        l_line = line.split()
        bool_1 = "variable" in line and "L" in line and "equal" in line
        bool_2 = len(l_line) == 4
        bool_3 = True
        if bool_2:
            try:
                m_l = float(l_line[3])
                if m_l > 1:
                    bool_3 = True
            except:
                bool_3 = False
        return bool_1 and bool_2 and bool_3

    def _run_print(self):
        """Print the FOM."""
        for key in self.metrics_cache:
            logger.info("{} = {}".format(key, self.metrics_cache[key]))

    def _run_csv(self):
        """Save the FOM to CSV."""
        file_csv = self.file_name + ".csv"
        with open(file_csv, "w", newline="") as csvfile:
            csvwriter = csv.writer(
                csvfile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            l_top = []
            l_bottom = []
            for key in self.metrics_cache:
                l_top.append(key)
                l_bottom.append(self.metrics_cache[key])
            csvwriter.writerow(l_top)
            csvwriter.writerow(l_bottom)

    def run(self):
        """Extract the FOM."""
        self.logger.debug("Extracting the FOM...")

        self.metrics_cache["FOM (M-particle-steps/sec/node)"] = None
        self.metrics_cache["No. Ranks"] = None
        self.metrics_cache["No. Nodes"] = None
        self.metrics_cache["Wall Time (sec)"] = None
        self.metrics_cache["No. Steps"] = None
        self.metrics_cache["No. Particles"] = None
        self.metrics_cache["Particles Per Cell [PPC]"] = None
        self.metrics_cache["Length Scaling Factor [L]"] = None
        self.metrics_cache["File"] = os.path.abspath(self.file_name)

        loop_info = []
        is_extract = False

        with open(self.file_name) as fp:
            cnt = 1
            line = fp.readline()
            while line:
                cnt += 1
                line = fp.readline()
                if self.metrics_cache["Particles Per Cell [PPC]"] is None:
                    if self._check_ppc(line):
                        l_line = line.split()
                        self.metrics_cache["Particles Per Cell [PPC]"] = int(l_line[3])
                if self.metrics_cache["Length Scaling Factor [L]"] is None:
                    if self._check_l(line):
                        l_line = line.split()
                        self.metrics_cache["Length Scaling Factor [L]"] = float(
                            l_line[3]
                        )
                if self.metrics_cache["No. Ranks"] is None:
                    if self._check_ranks(line):
                        l_line = line.split()
                        self.metrics_cache["No. Ranks"] = int(l_line[2])
                        self.metrics_cache["No. Nodes"] = int(
                            round(
                                self.metrics_cache["No. Ranks"]
                                / self.num_ranks_per_node
                            )
                        )
                        continue
                if self._check_end(line):
                    #    0    1  2       3  4    5     6   7    8     9   10          11        12
                    # Loop time of 784.094 on 7168 procs for 6866 steps with 16121689438 particles
                    self.logger.debug("Found end at line {}.".format(cnt))
                    l_line = line.split()
                    self.metrics_cache["Wall Time (sec)"] = float(l_line[3])
                    self.metrics_cache["No. Steps"] = int(l_line[8])
                    self.metrics_cache["No. Particles"] = int(l_line[11])
                    break
                if is_extract:
                    loop_info.append(self._extract_line(line))
                if self._check_start(line):
                    self.logger.debug("Found start at line {}.".format(cnt))
                    is_extract = True
                    continue
        self.metrics_cache["FOM (M-particle-steps/sec/node)"] = (
            self._compute_fom(loop_info) / self.metrics_cache["No. Nodes"]
        )
        if self.metrics_cache["Wall Time (sec)"] is not None:
            if self.metrics_cache["Wall Time (sec)"] < 600.0:
                self.metrics_cache["FOM (M-particle-steps/sec/node)"] = None
        else:
            self.metrics_cache["FOM (M-particle-steps/sec/node)"] = None

        self._run_print()
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
    sparta_fom = SpartaFom(
        logger=logger,
        file_name=cl_args.file,
        is_all=cl_args.all,
        num_ranks_per_node=cl_args.numRanksPerNode,
    )

    # do work
    sparta_fom.run()

    # exit gracefully
    sys.exit(EXIT_CODES["success"])
