from collections import OrderedDict
import pavilion.module_wrapper as module_wrapper
from pavilion.module_actions import ModuleLoad, ModuleSwap, ModuleUnload

class CraympichWrapper(module_wrapper.ModuleWrapper):
    def __init__(self):
        # This is a wrapper for cray-mpich, for any version.
        super().__init__(
            'cray-mpich',
            "Wrapper for the Cray-MPICH module",
            None,
            self.PRIO_COMMON)

    def load(self, var_man, requested_version=None):
        if requested_version is None:
            requested_version = self._version
        actions = list()
        env = OrderedDict()

        actions.append(ModuleLoad(self.name, version=requested_version))
        env['PAV_MPI_CC'] = '$(which cc)'
        env['PAV_MPI_CXX'] = '$(which CC)'
        env['PAV_MPI_FC'] = '$(which ftn)'
        env['PAV_MPI_RUN'] = '$(which mpiexec)'

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
