import sys
sys.setrecursionlimit(20_000)

#############################
# Scheme-related Exceptions #
#############################


class SchemeError(Exception):
    """
    A type of exception to be raised if there is an error with a Scheme
    program.  Should never be raised directly; rather, subclasses should be
    raised.
    """

    pass


class SchemeSyntaxError(SchemeError):
    """
    Exception to be raised when trying to evaluate a malformed expression.
    """

    pass


class SchemeNameError(SchemeError):
    """
    Exception to be raised when looking up a name that has not been defined.
    """

    pass


class SchemeEvaluationError(SchemeError):
    """
    Exception to be raised if there is an error during evaluation other than a
    SchemeNameError.
    """

    pass


############################
# Tokenization and Parsing #
############################


def number_or_symbol(value):
    """
    Helper function: given a string, convert it to an integer or a float if
    possible; otherwise, return the string itself
    """
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def tokenize(source):
    """
    Splits an input string into meaningful tokens (left parens, right parens,
    other whitespace-separated values).  Returns a list of strings.
    """
    lines = source.split("\n")
    total_tokens = []
    for line in lines:
        j = 0
        tokens = line
        for char in line:
            if char == ";":
                tokens = tokens[:j]
                break
            elif char in "()":
                tokens = tokens[:j] + " " + tokens[j] + " " + tokens[j + 1 :]
                j += 3
            else:
                j += 1
        total_tokens += tokens.split(" ")

    return [token for token in total_tokens if token != ""]


def parse(tokens):
    """
    Parses a list of tokens, constructing a representation where:
        * symbols are represented as Python strings
        * numbers are represented as Python ints or floats
        * S-expressions are represented as Python lists
    """
    def parse_expression(index):
        if tokens[index] == "(":
            search_index = index + 1
            exp_list = []

            try:
                while tokens[search_index] != ")":
                    exp, search_index = parse_expression(search_index)
                    exp_list.append(exp)
            except IndexError:
                raise SchemeSyntaxError

            return exp_list, search_index + 1

        elif tokens[index] == ")":
            raise SchemeSyntaxError

        else:
            return number_or_symbol(tokens[index]), index + 1

    possible_parse, final_index = parse_expression(0)
    if final_index < len(tokens):
        raise SchemeSyntaxError
    else:
        return possible_parse


######################
# Built-in Functions #
######################


def multiply(args):
    if len(args) == 1:
        return args[0]

    curr_val = args[0]
    for arg in args[1:]:
        curr_val *= arg
    return curr_val


def divide(args):
    if len(args) == 1:
        return args[0]

    curr_val = args[0]
    for arg in args[1:]:
        curr_val /= arg
    return curr_val


def not_func(arg):
    if len(arg) != 1:
        raise SchemeEvaluationError
    return not arg[0]


def car(X):
    if len(X) != 1:
        raise SchemeEvaluationError
    try:
        if not isinstance(X[0], Pair):
            raise SchemeEvaluationError
        return X[0].car
    except:
        raise SchemeEvaluationError


def cdr(X):
    if len(X) != 1:
        raise SchemeEvaluationError
    try:
        # print(X)
        if not isinstance(X[0], Pair):
            raise SchemeEvaluationError
        return X[0].cdr
    except:
        raise SchemeEvaluationError


def list_func(args):
    """
    Given a list of arguments, make a linked list with them
    """
    if not args:
        return scheme_builtins["[]"]
    return Pair(args[0], list_func(args[1:]))


def is_list(pos_list):
    """
    Given a list (ideally with a single element), determine if it
    constitutes a linked list

    takes in [Pair], [[]], [any# of args], or [any_element]
    """
    if len(pos_list) != 1:
        raise SchemeEvaluationError

    try:
        if pos_list == [scheme_builtins["[]"]]:
            return True
        if scheme_builtins["[]"] == cdr(pos_list):
            return True
        else:
            return is_list([cdr(pos_list)])
    except:
        return False


def list_length(pos_list):
    """
    Given a list (ideally with a single element), determine its length
    """
    if not is_list(pos_list):
        raise SchemeEvaluationError

    if pos_list == [scheme_builtins["[]"]]:
        return 0

    def recurse(pos_list):
        if cdr(pos_list) == scheme_builtins["[]"]:
            return 0

        return 1 + recurse([cdr(pos_list)])

    return 1 + recurse(pos_list)


def list_ref(list_and_index):
    """
    Given a list of length 2 (with a linked list and index), 
    determine the value at the specified index
    """
    if len(list_and_index) != 2:
        raise SchemeEvaluationError

    if not isinstance(list_and_index[0], Pair) or not isinstance(
        list_and_index[1], int
    ):
        raise SchemeEvaluationError

    if list_and_index[1] == 0:
        return car([list_and_index[0]])

    return list_ref([cdr([list_and_index[0]]), list_and_index[1] - 1])


def list_append(lists_arg):
    """
    Takes a list of linked lists, and concatenates them

    Does not mutate linked list arguments
    """
    val_list = []
    for list in lists_arg:
        length = list_length([list])  # will raise an error if not a list!
        for i in range(length):
            val_list.append(list_ref([list, i]))

    return list_func(val_list)


scheme_builtins = {
    "+": sum,
    "-": lambda args: -args[0] if len(args) == 1 else (args[0] - sum(args[1:])),
    "*": multiply,
    "/": divide,
    "equal?": lambda args: (
        True if len(args) == 1 else all(args[0] == args[i] for i in range(1, len(args)))
    ),
    ">": lambda args: (
        True
        if len(args) == 1
        else all(args[i] > args[i + 1] for i in range(len(args) - 1))
    ),
    ">=": lambda args: (
        True
        if len(args) == 1
        else all(args[i] >= args[i + 1] for i in range(len(args) - 1))
    ),
    "<": lambda args: (
        True
        if len(args) == 1
        else all(args[i] < args[i + 1] for i in range(len(args) - 1))
    ),
    "<=": lambda args: (
        True
        if len(args) == 1
        else all(args[i] <= args[i + 1] for i in range(len(args) - 1))
    ),
    "not": not_func,
    "#t": True,
    "#f": False,
    "[]": "emptylist!",
    "car": car,
    "cdr": cdr,
    "list": list_func,
    "list?": is_list,
    "length": list_length,
    "list-ref": list_ref,
    "append": list_append,
    "begin": lambda args: args[-1],
}


##############
# Evaluation #
##############


class Frame:
    """
    Creates a frame that can store variables

    Takes a parent frame upon initialization. During variable lookup,
    if variable is not present, then looks in parent frame, and so on
    """

    def __init__(self, parent=scheme_builtins):
        if parent == scheme_builtins:
            self.parent = scheme_builtins
        else:
            self.parent = parent

        self.vals = {}

    def __getitem__(self, key):
        try:
            return self.vals[key]
        except:
            try:
                return self.parent[key]
            except:
                raise SchemeNameError

    def __setitem__(self, key, value):
        self.vals[key] = value

    def update_item(self, key, value):
        try:
            _ = self.vals[key]
            self.vals[key] = value
        except:
            try:
                self.parent.update_item(key, value)
            except:
                raise SchemeNameError


class Function:
    """
    Creates a function that takes in a list of parameters and returns
    a value.

    Takes as argument the frame which it was initialized in as well as
    a list of parameters
    """

    def __init__(self, pars, exp, frame):
        self.pars = pars
        self.exp = exp
        self.enclose_frame = frame

    def func_eval(self, eval_args):
        """
        Completely evaluates the function, returns a number
        """
        scope = Frame(self.enclose_frame)
        if len(eval_args) != len(self.pars):
            raise SchemeEvaluationError

        if eval_args:
            for index, par in enumerate(self.pars):
                scope[par] = eval_args[index]

        return evaluate(self.exp, scope)


class Pair:
    """
    Creates a cons cell!
    """

    def __init__(self, car, cdr):
        self.car = car
        self.cdr = cdr


def make_initial_frame():
    return Frame()


def evaluate(tree, frame=make_initial_frame()):
    """
    Evaluate the given syntax tree according to the rules of the Scheme
    language.
    """
    tree_location = tree

    # single value expression base cases
    if isinstance(tree, (float, int, Pair)):
        return tree
    elif isinstance(tree, str):
        try:
            return frame[tree]
        except:
            raise SchemeNameError
    elif tree == []:
        return frame["[]"]

    # lambda special case (returns a function)
    if tree_location[0] == "lambda":
        return Function(tree_location[1], tree_location[2], frame).func_eval

    # and / or special case (returns a boolean)
    if tree_location[0] == "and":
        for argument in tree_location[1:]:
            if argument == "#f" or not evaluate(argument, frame):
                return False
        return True
    if tree_location[0] == "or":
        for argument in tree_location[1:]:
            if argument == "#t" or evaluate(argument, frame):
                return True
        return False

    # if special case (returns a boolean)
    if tree_location[0] == "if":
        if evaluate(tree_location[1], frame):
            return evaluate(tree_location[2], frame)
        else:
            return evaluate(tree_location[3], frame)

    # cons special case
    if tree_location[0] == "cons":
        if len(tree_location[1:]) != 2:
            raise SchemeEvaluationError

        return Pair(
            evaluate(tree_location[1], frame), evaluate(tree_location[2], frame)
        )

    # del special case
    if tree_location[0] == "del":
        if len(tree_location[1:]) != 1:
            raise SchemeEvaluationError
        try:
            val = frame.vals[tree_location[1]]
            del frame.vals[tree_location[1]]
            return val
        except:
            raise SchemeNameError

    # let special case (returns an evaluation)
    if tree_location[0] == "let":
        try:
            temp_frame = Frame(frame)
            for declaration in tree_location[1]:
                val = evaluate(declaration[1], frame)
                temp_frame[declaration[0]] = val
            return evaluate(tree_location[2], temp_frame)
        except SchemeNameError:
            raise SchemeNameError
        except:
            raise SchemeEvaluationError

    # set! special case (returns variable value it just set)
    if tree_location[0] == "set!":
        exp = evaluate(tree_location[2], frame)
        frame.update_item(tree_location[1], exp)
        return exp

    # define special case (returns string "function object" or val)
    if tree_location[0] == "define":
        if isinstance(tree_location[1], list):
            name = tree_location[1][0]
            parameters = tree_location[1][1:]
            tree_location = ["lambda"] + [parameters] + [tree_location[2]]
            frame[name] = evaluate(tree_location, frame)
        else:
            name = tree_location[1]
            frame[name] = evaluate(tree_location[2], frame)
        if isinstance(frame[name], (int, float, Pair)):
            return frame[name]
        else:
            return "function object"

    # general case (finds a func, evaluates args)
    if not isinstance(tree_location[0], (int, float)):
        func = evaluate(tree_location[0], frame)
    else:
        raise SchemeEvaluationError
    args = []
    for el in tree_location[1:]:
        args.append(evaluate(el, frame))

    return func(args)


def evaluate_file(file_name, frame=make_initial_frame()):
    with open(file_name, "r") as file:
        file_content = file.read()  # currently in scheme format
        tokens = tokenize(file_content)
        parsed_exp = parse(tokens)
        return evaluate(parsed_exp, frame)


if __name__ == "__main__":
    g_frame = make_initial_frame()
    for file_name in sys.argv[1:]:
        evaluate_file(file_name, g_frame)
    # for evaluating scheme in different files via the command line

    pass
