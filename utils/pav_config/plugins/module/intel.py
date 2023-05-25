from collections import OrderedDict
import pavilion.module_wrapper as module_wrapper
from pavilion.module_actions import ModuleLoad, ModuleSwap, ModuleUnload

class IntelWrapper(module_wrapper.ModuleWrapper):
    def __init__(self):
        # This is a wrapper for intel, for any version.
        super().__init__('intel',
                         "Wrapper for the Intel module",
                         None,
                         self.PRIO_COMMON)

    def load(self, var_man, requested_version=None):
        if requested_version is None:
            requested_version = self._version
        actions = list()
        env = OrderedDict()

        if 'cle' in var_man['sys_os.name']:
            if var_man['sys_arch'] == 'aarch64':
                actions.append(ModuleSwap('PrgEnv-intel', '', 'PrgEnv-cray', ''))
            actions.append(ModuleSwap(self.name, '', self.name,
                                      requested_version))
            # Prepending with PAV_ to differentiate from these variables that
            # cray already uses.
            env['PAV_CC'] = '$(which cc)'
            env['PAV_CXX'] = '$(which CC)'
            env['PAV_FC'] = '$(which ftn)'
        else:
            actions.append(ModuleLoad(self.name, version=requested_version))
            # Prepending with PAV_ for consistency.
            env['PAV_CC'] = '$(which icc)'
            env['PAV_CXX'] = '$(which icpc)'
            env['PAV_FC'] = '$(which ifort)'

        return actions, env

    def remove(self, var_man, requested_version=None):
        actions = list()
        env = OrderedDict()

        version = self.get_version(requested_version)

        env['PAV_CC'] = ''
        env['PAV_CXX'] = ''
        env['PAV_FC'] = ''

        actions.append(ModuleUnload(self.name, version))
        if 'cle' in var_man['sys_os.name']:
            actions.append(ModuleUnload('PrgEnv-intel', ''))

        return actions, env
