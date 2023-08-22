from collections import OrderedDict
import pavilion.module_wrapper as module_wrapper
from pavilion.module_actions import ModuleLoad, ModuleSwap, ModuleUnload

class GccWrapper(module_wrapper.ModuleWrapper):
    def __init__(self):
        # This is a wrapper for gcc, for any version.
        super().__init__('gcc',
                         "Wrapper for the GCC module",
                         None,
                         self.PRIO_COMMON)

    def load(self, var_man, requested_version=None):
        if requested_version is None:
            requested_version = self._version
        actions = list()
        env = OrderedDict()

        if var_man['sys_os.name'].startswith('cle'):
            if var_man['sys_arch'] == 'aarch64':
                actions.append(ModuleSwap('PrgEnv-gnu', '', 'PrgEnv-cray', ''))
            else:
                actions.append(ModuleSwap('PrgEnv-gnu', '', 'PrgEnv-intel', ''))
            actions.append(ModuleSwap(self.name, requested_version, self.name,
                                      ''))
            env['PAV_CC'] = '$(which cc)'
            env['PAV_CXX'] = '$(which CC)'
            env['PAV_FC'] = '$(which ftn)'
        else:
            actions.append(ModuleLoad(self.name, version=requested_version))
            env['PAV_CC'] = '$(which gcc)'
            env['PAV_CXX'] = '$(which g++)'
            env['PAV_FC'] = '$(which gfortran)'

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
            actions.append(ModuleUnload('PrgEnv-gnu', ''))

        return actions, env
