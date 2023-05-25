from collections import OrderedDict
import pavilion.module_wrapper as module_wrapper
from pavilion.module_actions import ModuleLoad, ModuleSwap, ModuleUnload

class IntelmpiWrapper(module_wrapper.ModuleWrapper):
    def __init__(self):
        # This is a wrapper for intel-mpi, for any version.
        super().__init__(
            'intel-mpi',
            "Wrapper for the Intel-MPI module",
            None,
            self.PRIO_COMMON)

    def load(self, var_man, requested_version=None):
        if requested_version is None:
            requested_version = self._version
        actions = list()
        env = OrderedDict()

        actions.append(ModuleLoad(self.name, version=requested_version))
        env['PAV_MPI_CC'] = '$(which mpiicc)'
        env['PAV_MPI_CXX'] = '$(which mpiicpc)'
        env['PAV_MPI_FC'] = '$(which mpiifort)'
        env['PAV_MPI_RUN'] = '$(which mpirun)'

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
