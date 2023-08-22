from collections import OrderedDict
import pavilion.module_wrapper as module_wrapper
from pavilion.module_actions import ModuleLoad, ModuleSwap, ModuleUnload

class Mvapich2Wrapper(module_wrapper.ModuleWrapper):
    def __init__(self):
        # This is a wrapper for mvapich2, for any version.
        super().__init__(
            'mvapich2',
            "Wrapper for the Mvapich2 module",
            None,
            self.PRIO_COMMON)

    def load(self, var_man, requested_version=None):
        if requested_version is None:
            requested_version = self._version
        actions = list()
        env = OrderedDict()

        actions.append(ModuleLoad(self.name, version=requested_version))
        env['PAV_MPI_CC'] = '$(which mpicc)'
        env['PAV_MPI_CXX'] = '$(which mpicxx)'
        env['PAV_MPI_FC'] = '$(which mpifort)'
        env['PAV_MPI_RUN'] = ''

        return actions, env

    def remove(self, var_man, requested_version=None):
        actions = list()
        env = OrderedDict()

        version = self.get_version(requested_version)

        env['PAV_MPI_CC'] = ''
        env['PAV_MPI_CXX'] = ''
        env['PAV_MPI_FC'] = ''
        env['PAV_MPI_RUN'] = ''

        actions.append(ModuleUnload(self.name, version))

        return actions, env
