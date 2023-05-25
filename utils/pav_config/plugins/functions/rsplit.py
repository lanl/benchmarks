from pavilion import expression_functions as funcs


class RSplit(funcs.FunctionPlugin):
    """Split a string using the given seperator, up to max_splits times, 
    from right to left.
    Max_splits of zero splits unlimited times. 
    An empty string as the seperator splits on whitespace."""

    def __init__(self):
        """Initialize the plugin."""

        super().__init__(
            name='rsplit',
            arg_specs=(str, str, int)
        )

    def rsplit(self, string, sep, max_splits):
        """Just uses the string rsplit method."""

        if sep == '':
            sep = None

        if max_splits <= 0:
            max_splits = -1

        return string.rsplit(sep, max_splits)
