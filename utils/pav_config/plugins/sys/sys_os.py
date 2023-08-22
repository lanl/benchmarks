from pavilion import sys_vars
from pathlib import Path
import subprocess


class SystemOS(sys_vars.SystemPlugin):

    def __init__(self):
        super().__init__(
            name='sys_os',
            description="The LANL HPC system os info (name, version).",
            priority=20,
            is_deferable=False,
            sub_keys=['name', 'version'])

    def _get(self):
        """Base method for determining the operating system and version."""

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
        elif 'cos' in os_all:
            os['name'] = 'cos'
            os['version'] = os_all[3:]
        elif 'darwin' in os_all:
            os['name'] = 'darwin'
            os['version'] = 'none'

        return os
