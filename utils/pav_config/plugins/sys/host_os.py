from pavilion import sys_vars
from pathlib import Path
import subprocess


class HostOS(sys_vars.SystemPlugin):

    def __init__(self):
        super().__init__(
        name='host_os',
        description="The target LANL HPC host's OS info (name, version).",
        priority=20,
        is_deferable=True,
        sub_keys=['name', 'version'])

    def _get(self):
        """Base method for determining the operating host and version."""

        os = {}

        try:
            os_all = subprocess.check_output(
                ['/usr/projects/hpcsoft/utilities/bin/sys_os']
                ).strip().decode('UTF-8')
        except:
            os_all = subprocess.check_output(
                ['/projects/hpcsoft/utilities/bin/sys_os']
                ).strip().decode('UTF-8')

        if 'toss' in os_all:
            os['name'] = 'toss'
            os['version'] = os_all[-1]
        elif 'cle' in os_all:
            os['name'] = 'cle'
            os['version'] = os_all[3:]

        return os
