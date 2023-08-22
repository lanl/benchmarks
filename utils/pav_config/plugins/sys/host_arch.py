import subprocess
from pavilion import sys_vars

class HostArch(sys_vars.SystemPlugin):

    def __init__(self):
        super().__init__(
            name='host_arch',
            description="The current LANL HPC host's architecture.",
            priority=20,
            is_deferable=True,
            sub_keys=None)

    def _get(self):
        """Base method for determining the host architecture."""

        try:
            out = subprocess.check_output(
                ['/usr/projects/hpcsoft/utilities/bin/sys_arch'])
        except:
            out = subprocess.check_output(
                ['/projects/hpcsoft/utilities/bin/sys_arch'])

        return out.strip().decode('utf8')

