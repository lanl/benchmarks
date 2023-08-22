import subprocess
from pavilion import sys_vars

class SystemArch(sys_vars.SystemPlugin):

    def __init__(self):
        super().__init__(
            name='sys_arch',
            description="The LANL HPC system architecture.",
            priority=20,
            is_deferable=False,
            sub_keys=None)

    def _get(self):
        """Base method for determining the system architecture."""

        try:
            arch = subprocess.check_output(
                ['/usr/projects/hpcsoft/utilities/bin/sys_arch'])
        except:
            arch = subprocess.check_output(
                ['/projects/hpcsoft/utilities/bin/sys_arch'])

        return arch.strip().decode('utf8')
