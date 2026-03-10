import ast

from pavilion import expression_functions as funcs


class Outliers2(funcs.FunctionPlugin):
    "Returns number of spikes."

    def __init__(self):
        """Initialize the plugin."""

        super().__init__(
            name='outliers2',
            arg_specs=(str, float)
        )

    def outliers2(self, all_latencies, limit):
        """Returns number of spikes."""

        # all_latencies is a string of a list of list of floats
        all_latencies = ast.literal_eval(all_latencies)
        num_spikes = 0

        for latencies in all_latencies:
            mean = sum(latencies) / len(latencies)
            stddev = (sum([(val - mean)**2 for val in latencies]) \
                /len(latencies))**0.5

            # keeping this variable here even though I can just increment
            # num_spikes directly in case we want outliers printed at one point
            outliers = list()
            for latency in latencies:
                dev = abs(latency - mean)/stddev
                if dev > limit:
                    outliers.append(latency)

            num_spikes += len(outliers)

        return num_spikes
