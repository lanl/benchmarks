from pavilion import expression_functions
# The 'num' function will accept any numerical looking type generically.
from pavilion.expression_functions import num

# All function plugins inherit from the 'FunctionPlugin' class
class Square(expression_functions.FunctionPlugin):

    # As with other plugin types, we override __init__ to provide the
    # basic information on our plugin.
    def __init__(self):

        super().__init__(
            # The name of our plugin and function
            name="square",

            # The short description shown when listing these plugins.
            description="Square the provided number",

            # The arg_specs define how to auto-convert arguments to the
            # appropriate types. More on that below.
            # Note: (foo,) is a single item tuple containing foo.
            arg_specs=(num,)
        )

    # This method is the 'function' this plugin defines. It should take
    # arguments as defined by the arg_spec. It should also return one of the
    # types understood by Pavilion expressions (int, float, bool, string, or
    # lists/dicts containing only those types).
    @staticmethod
    def square(num):
        """The docstring of the function will serve as its long
        documentation."""

        # We don't need to do any type checking, those conversions
        # will already be done for us (and will raise the appropriate
        # errors).
        return pow(num, 2)
