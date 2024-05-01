import ast


def type_infer(s):
    """ String to variable with type inference

    NOTE that yaml 1.2 has true, false, and null keywords,
    with all upper case, all lower case, or capital; e.g.) True, TRUE, true.

    If you want to get the string of the keywords, like 'True' or 'None', use escaping:
    e.g.) --key1.key2 \'true\' => 'true'
          --key1.key2 "'true'" => 'true'
    """
    def capitalize_ul(s):
        """ Capitalize string only if s is upper or lower case """
        if s.isupper() or s.islower():
            return s.capitalize()
        return s

    try:
        # Special case
        # 1. Treat 'None' as 'None' string, not None keyword
        if s.strip() == 'None':
            s = '\'None\''

        ts = capitalize_ul(s)
        if ts == 'True':
            s = 'True'
        elif ts == 'False':
            s = 'False'
        elif ts == 'Null':
            s = 'None'

        s = ast.literal_eval(s)
        return s
    except Exception:
        return s
