from pavilion import expression_functions as funcs


class Idx(funcs.FunctionPlugin):
    """Return the nth item from the given list. Reverse
    (negative) indexes are allowed."""

    def __init__(self):
        """Initialize the plugin."""

        super().__init__(
            name='idx',
            arg_specs=([None], int)
        )

    def idx(self, ilist, idx):
        """Return the requested item. Asking for an item out of 
        range is an error."""

        try:
            return ilist[idx]
        except IndexError:
            raise funcs.FunctionPluginError(
               "Index '{}' is out of range for list '{}' in "
               "the idx() function.".format(idx, ilist))

