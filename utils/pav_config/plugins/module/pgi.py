from collections import OrderedDict
import pavilion.module_wrapper as module_wrapper
from pavilion.module_actions import ModuleLoad, ModuleSwap, ModuleUnload

class PgiWrapper(module_wrapper.ModuleWrapper):
    def __init__(self):
        # This is a wrapper for pgi, for any version.
        super().__init__('pgi',
                         "Wrapper for the PGI module",
                         None,
                         self.PRIO_COMMON)

    def load(self, var_man, requested_version=None):
        if requested_version is None:
            requested_version = self._version
        actions = list()
        env = OrderedDict()

        actions.append(ModuleLoad(self.name, version=requested_version))
        # Prepending with PAV_ for consistency.
        env['PAV_CC'] = '$(which pgcc)'
        env['PAV_CXX'] = '$(which pgc++)'
        env['PAV_FC'] = '$(which pgfortran)'

        return actions, env

    def remove(self, var_man, requested_version=None):
        actions = list()
        env = OrderedDict()

        version = self.get_version(requested_version)

        env['PAV_CC'] = ''
        env['PAV_CXX'] = ''
        env['PAV_FC'] = ''

        actions.append(ModuleUnload(self.name, version))

        return actions, env
