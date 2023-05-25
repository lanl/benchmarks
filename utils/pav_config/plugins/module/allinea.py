from collections import OrderedDict
import pavilion.module_wrapper as module_wrapper
from pavilion.module_actions import ModuleLoad, ModuleSwap, ModuleUnload

class AllineaWrapper(module_wrapper.ModuleWrapper):
    def __init__(self):
        # This is a wrapper for allinea compiler, for any version.
        super().__init__('allinea',
                         "Wrapper for the Allinea module",
                         None,
                         self.PRIO_COMMON)

    def load(self, var_man, requested_version=None):
        if requested_version is None:
            requested_version = self._version
        actions = list()
        env = OrderedDict()

        if var_man['sys_arch'] == 'aarch64':
            actions.append(ModuleSwap('PrgEnv-allinea', '', 'PrgEnv-cray', ''))
        else:
            actions.append(ModuleSwap('PrgEnv-allinea', '', 'PrgEnv-intel', ''))
        actions.append(ModuleSwap(self.name, '', self.name, requested_version))
        # Prepending with PAV_ to differentiate from these variables that
        # cray already uses.
        env['PAV_CC'] = '$(which cc)'
        env['PAV_CXX'] = '$(which CC)'
        env['PAV_FC'] = '$(which ftn)'

        return actions, env

    def remove(self, var_man, requested_version=None):
        actions = list()
        env = OrderedDict()

        version = self.get_version(requested_version)

        env['PAV_CC'] = ''
        env['PAV_CXX'] = ''
        env['PAV_FC'] = ''

        actions.append(ModuleUnload(self.name, version))
        actions.append(ModuleUnload('PrgEnv-allinea', ''))

        return actions, env
