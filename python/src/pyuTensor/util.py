import inspect
from types import ModuleType

__all__ = ["export"]


def export(obj):
    """export object

    populate the given object into `__all__`
    """
    caller_frame = inspect.currentframe().f_back
    all_list = caller_frame.f_globals.get("__all__", [])
    caller_frame.f_globals["__all__"] = all_list
    try:
        all_list.append(
            obj.__name__.split(".")[-1] if isinstance(obj, ModuleType) else obj.__name__
        )  # exporting a class/function
    except AttributeError as err:
        for name, value in caller_frame.f_globals.items():
            if value is obj:
                all_list.append(name)
                break
        else:
            raise ValueError(
                f"{obj} not found in the scope, fail to export symbol"
            ) from err
    return obj