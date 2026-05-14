from pavilion import result_parsers
from pavilion import pav_vars
from pavilion import system_variables
import yaml_config as yc
from datetime import datetime
import re
import json

class splunk_stream(result_parsers.ResultParser):
    """Collects appropriate test data for splunk."""

    PASS = result_parsers.PASS
    FAIL = result_parsers.FAIL

    def __init__(self):
        super().__init__(name='splunk_stream')

    def get_config_items(self):
        config_items = super().get_config_items()
        config_items.extend([
            yc.StrElem(
                'search', default=None,
                help_text="A word that will be searched for in the output of "
                          "the test that will determine success or failure."
            ),
            yc.KeyedElem(
                'data', elements=[], default={},
                help_text="Information to pull out."
            )
        ])

        return config_items

    def check_args(self, test, file=None, search=None, data=None):

        if search == "":
            raise result_parsers.ResultParserError(
                "Splunk_Stream result parser requires a non emtpy string for searching."
            )

    def __call__(self, test, file=None, search=None, data=None):

        found = False

        self.data = {
            "WriteMax(MiB)" : "bwMaxMIB",
            "WriteMin(MiB)" : "bwMinMIB",
            "WriteMean(MiB)" : "bwMeanMIB",
            "WriteStdDev" : "bwStdMIB",
            "ReadMax(MiB)" : "bwMaxMIB",
            "ReadMin(MiB)" : "bwMinMIB",
            "ReadMean(MiB)" : "bwMeanMIB",
            "ReadStdDev" : "bwStdMIB"
          }

        try:
            if search in open(file).read():
                found = True
        except (IOError, OSError) as err:
            raise result_parsers.ResultParserError(
                "Splunk_Stream result parser could not read input file '{}': {}".format(file, err)
                .format(file, err)
            )

        if found:
            #USED TO REMOVE EVERYTHING THAT ISN'T JSON
            with open(file, 'r') as run_log:
                d = run_log.readlines()

            with open("data.json", 'w') as write_log:
               if 'ior WARNING' in d[7]:
                   write_log.writelines(d[8:])
               else:
                   write_log.writelines(d[7:])

            with open("data.json") as json_file:
                d = json.load(json_file)

            write_data = d["summary"][0]
            read_data = d["summary"][1]

            for key in self.data.keys():
                if "Write" in key:
                    self.data[key] = write_data[self.data[key]]
                else:
                    self.data[key] = read_data[self.data[key]]

            return self.data

        else:
            return self.FAIL

