from pavilion import expression_functions
from pavilion.expression_functions import num

class IntConvPlugin(expression_functions.FunctionPlugin):
    """Convert integer strings to ints of arbitrary bases."""

    def __init__(self):
        """Setup plugin."""

        super().__init__(
            name="intconv",
            description="Covert to type 'int'",
            arg_specs=(num, num),
        )

    @staticmethod
    def intconv(value, base):
        """Convert the given string 'value' as an integer of
         the given base. Bases from 2-32 are allowed."""

        return int(value, base)
