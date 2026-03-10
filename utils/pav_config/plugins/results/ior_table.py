from pavilion import result_parsers
import re


class IORTable(result_parsers.ResultParser):

    """Parses tables."""

    def __init__(self):
        super().__init__(
            name='ior_table',
            description="Parses IOR summary table."
        )

    def __call__(self, test, file, delimiter=None):

        match_list = []

        # IOR table regular expression
        value_and_space = '(\S+)\s+'
        str_regex = value_and_space * 25
        str_regex = '^\s?' + str_regex

        regex = re.compile(str_regex)
        for line in file.readlines():
            match_list.extend(regex.findall(line))

        # first element of match_list contains column names
        col_names = list(match_list.pop(0))
        col_names.pop(0)

        result_dict = {}
        row_names = []
        for m_idx in range(len(match_list)):
            row_names.append(match_list[m_idx][0])
            match_list[m_idx] = match_list[m_idx][1:]

        for col_idx in range(len(col_names)):

            # replace "special" characters
            cur_col_name = col_names[col_idx]
            cur_col_name = cur_col_name.replace("(", "_")
            cur_col_name = cur_col_name.replace(")", "_")
            cur_col_name = cur_col_name.replace("#", "Num")

            result_dict[cur_col_name] = {}
            for row_idx in range(len(row_names)):
                result_dict[cur_col_name][row_names[row_idx]] = \
                    match_list[row_idx][col_idx]

        return result_dict
