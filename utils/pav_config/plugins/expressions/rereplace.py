from pavilion import expression_functions
from pavilion.expression_functions import num
import re

class ReReplacePlugin(expression_functions.FunctionPlugin):
    """Replace a string that matches a regular expression."""

    def __init__(self):
        """Setup plugin."""

        super().__init__(
            name="rereplace",
            description="Replace regex matched string",
            arg_specs=(str, str, str, int),
        )

    @staticmethod
    def rereplace(pattern, replacement, original, count):
        """Replace the regular expression matched string with the string
        provided."""

        return re.sub(pattern, replacement, original, count)
