import subprocess
from pavilion import sys_vars


class HostName(sys_vars.SystemPlugin):

    def __init__(self):
        super().__init__(
        name='host_name',
        description="The target LANL HPC host's hostname.",
        priority=20,
        is_deferable=True,
        sub_keys=None)

    def _get(self):
        """Base method for determining the host name."""

        try:
            out = subprocess.check_output(
                ['/usr/projects/hpcsoft/utilities/bin/sys_name'])
        except:
            out = subprocess.check_output(
                ['/projects/hpcsoft/utilities/bin/sys_name'])

        return out.strip().decode('UTF-8')

