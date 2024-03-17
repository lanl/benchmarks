#!/usr/bin/env python3

"""
This is a self-contained script that computes SPARTA run statistics from a
large, repeating ensemble.

This self-contained script extracts the min, max, average, and std. dev. for
the specified column header.
Author: Anthony M. Agelastos <amagela@sandia.gov>
"""


# import Python functions
import sys
import argparse
import os
import logging
import csv
import statistics

assert sys.version_info >= (3, 5), "Please use Python version 3.5 or later."


# define GLOBAL vars
VERSION = "2.71"
TIMEOUT = 30
IS_ALL = True
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
            description="This Python program will extract the figure of merit (FOM) statistics for SPARTA.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            epilog=my_epilog,
        )

        self.parser.add_argument(
            "-a",
            "--all",
            action="store_true",
            default=IS_ALL,
            help="Generate ALL FOM stat information",
        )

        self.parser.add_argument(
            "-f",
            "--file",
            type=str,
            default="metrics_rollup.csv",
            help="file name to read",
        )

        self.parser.add_argument(
            "-m",
            "--figureOfMerit",
            type=str,
            default="FOM (M-particle-steps/sec/node)",
            help="header to compute statistics for",
        )

        self.parser.add_argument(
            "-r",
            "--ranks",
            type=str,
            default="No. Ranks",
            help="header that stores the no. of MPI ranks",
        )

        self.parser.add_argument(
            "-p",
            "--PPC",
            type=str,
            default="Particles Per Cell [PPC]",
            help="header that stores the particles per cell (PPC) information",
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


class SpartaFomStats(object):
    """This class encapsulates computing the stats from SPARTA ensembles."""

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
            "header_fom",
            "header_ppc",
            "header_ranks",
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

    def _get_data(self):
        """Get the data."""
        self.logger.debug("Reading the CSV...")

        col_fom = None
        col_ppc = None
        col_ranks = None
        self.cache_fom = {}
        with open(self.file_name, newline="") as csvfile:
            csvfilereader = csv.reader(csvfile, delimiter=",", quotechar='"')
            for row in csvfilereader:
                if col_fom is None:
                    try:
                        col_fom = row.index(self.header_fom)
                    except ValueError:
                        pass
                if col_ppc is None:
                    try:
                        col_ppc = row.index(self.header_ppc)
                    except ValueError:
                        pass
                if col_ranks is None:
                    try:
                        col_ranks = row.index(self.header_ranks)
                    except ValueError:
                        pass
                if None in (col_fom, col_ppc, col_ranks):
                    continue
                if str(row[col_fom]) == self.header_fom:
                    continue

                key = (
                    str(row[col_ppc]).zfill(3)
                    + "|"
                    + str(row[col_ranks]).zfill(3)
                )
                self.logger.debug("key = {}".format(key))
                if key not in self.cache_fom:
                    self.cache_fom[key] = []
                try:
                    self.cache_fom[key].append(float(row[col_fom]))
                except ValueError:
                    self.logger.critical("Bad value: '{}':'{}' (column {})".format(key, row[col_fom], col_fom))

    def _compute_stats(self):
        """Compute the stats."""
        csvout_line = ["Key", "Min", "Mean", "Max", "Stdev", "Range", "Variability"]
        csvout_file_name = self.file_name.replace(".csv", "") + "--stats.csv"
        with open(csvout_file_name, "w", newline="") as csvoutfile:
            cofw = csv.writer(
                csvoutfile,
                delimiter=",",
                quotechar='"',
                quoting=csv.QUOTE_NONNUMERIC,
            )
            cofw.writerow(csvout_line)
            for key, value in self.cache_fom.items():
                csvout_line = []
                csvout_line.append(key)  # Key
                m_min = min(value)
                csvout_line.append(m_min)  # Min
                m_mean = statistics.mean(value)
                csvout_line.append(m_mean)  # Mean
                m_max = max(value)
                csvout_line.append(m_max)  # Max
                csvout_line.append(statistics.stdev(value))  # Stdev
                csvout_line.append(m_max - m_min)  # Range
                csvout_line.append((m_max - m_min) / m_mean)  # Variability
                cofw.writerow(csvout_line)

    def run(self):
        """Extract the stats."""
        self._get_data()
        self._compute_stats()


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
    sparta_fom_stats = SpartaFomStats(
        logger=logger,
        file_name=cl_args.file,
        is_all=cl_args.all,
        header_fom=cl_args.figureOfMerit,
        header_ppc=cl_args.PPC,
        header_ranks=cl_args.ranks,
    )

    # do work
    sparta_fom_stats.run()

    # exit gracefully
    sys.exit(EXIT_CODES["success"])
