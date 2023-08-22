import subprocess
from pavilion import sys_vars


class SystemName(sys_vars.SystemPlugin):

    def __init__(self):
        super().__init__(
            name='sys_name',
            description='The LANL HPC system name (not necessarily hostname).',
            priority=20,
            is_deferable=False,
            sub_keys=None)

    def _get(self):
        """Base method for determining the system name."""

        try:
            name = subprocess.check_output(
                ['/usr/projects/hpcsoft/utilities/bin/sys_name'])
        except:
            name = subprocess.check_output(
                ['/projects/hpcsoft/utilities/bin/sys_name'])

        return name.strip().decode('UTF-8')
