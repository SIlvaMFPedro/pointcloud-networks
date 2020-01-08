# ------------------------
#   IMPORTS
# ------------------------
# Import the necessary packages
import


def _variable_on_cpu(name, shape, initializer, use_fp16=False):
    """
        Helper to create a variable stored on CPU memory
        :param name: name of the variable
        :param shape: list of ints
        :param initializer: initializer for the variable
        :param use_fp16: boolean for fp16 usage
        :return: variable tensor
    """
