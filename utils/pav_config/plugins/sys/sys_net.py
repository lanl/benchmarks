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

        # sys_network script doesn't work on Venado for some reason
        name = subprocess.check_output([
            '/usr/projects/hpcsoft/utilities/bin/sys_name'])
        name = name.strip().decode('UTF-8')
        if name == 'venado':
            return 'red'
        else:
            network = subprocess.check_output([
                '/usr/projects/hpcsoft/utilities/bin/sys_network'])
            return network.strip().decode('utf8')
