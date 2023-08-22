import subprocess
import pavilion.sys_vars as sys_vars

class SystemNetwork(sys_vars.SystemPlugin):

    def __init__(self):
        super().__init__(
            name='sys_net',
            description="The LANL HPC system network.",
            priority=20,
            is_deferable=False,
            sub_keys=None)

    def _get(self):
        """Base method for determining the system network."""

        try:
            network = subprocess.check_output(
                ['/usr/projects/hpcsoft/utilities/bin/sys_network'])
        except:
            network = subprocess.check_output(
                ['/projects/hpcsoft/utilities/bin/sys_network'])

        return network.strip().decode('utf8')
