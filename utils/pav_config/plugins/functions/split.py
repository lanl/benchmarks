from pavilion import expression_functions as funcs


class Split(funcs.FunctionPlugin):
    """Split a string using the given seperator, up to max_splits times. 
    Max_splits of zero splits unlimited times. 
    An empty string as the seperator splits on whitespace."""

    def __init__(self):
        """Initialize the plugin."""

        super().__init__(
            name='split',
            arg_specs=(str, str, int)
        )

    def split(self, string, sep, max_splits):
        """Just use the string split method."""

        if sep == '':
            sep = None

        if max_splits <= 0:
            max_splits = -1

        return string.split(sep, max_splits)
