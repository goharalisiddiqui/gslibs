def check_int(non_negative: bool = False, **values):
        """Checks if the provided values are integers."""
        for name, value in values.items():
            if not isinstance(value, int):
                raise ValueError(f"Expected integer for {name}, got {type(value)} instead.")
            if non_negative and value < 0:
                raise ValueError(f"Expected non-negative integer for {name}, got {value} instead.")
    
def check_mutually_exclusive(require_one = False, **values):
    """Checks that at most one of the provided values is not None."""
    not_none_count = sum(1 for value in values.values() if \
        value is not None and \
        value is not False and \
        value != 0)
    if not_none_count > 1:
        raise ValueError(f"Expected at most one of {list(values.keys())} to be not None, "
                            f"but got {not_none_count} non-None values.")
    if require_one and not_none_count == 0:
        raise ValueError(f"Expected at least one of {list(values.keys())} to be used")

def check_exists(**paths):
    """Checks if the provided file paths exist."""
    import os
    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"File for {name} not found at path: {path}")

def check_limits(value: int, min_value: int = None, max_value: int = None):
    """Checks if the provided value is within the specified limits."""
    if min_value is not None and value < min_value:
        raise ValueError(f"Value {value} is less than minimum allowed {min_value}.")
    if max_value is not None and value > max_value:
        raise ValueError(f"Value {value} is greater than maximum allowed {max_value}.")