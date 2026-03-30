__all__ = [
    'DIY_dataset',
    'Function',
    'NN_model',
    'Plot_command',
]

import TopoDataSyn
from importlib import import_module

# Please keep this list sorted
assert __all__ == sorted(__all__)


def get_all_timeversions():
    results = {}
    for module_name in TopoDataSyn.__all__:
        try:
            module = import_module(f"{TopoDataSyn.__name__}.{module_name}")

            version_class = getattr(module, "version_info", None)
            if version_class is None:
                print(f"{module_name} 中没有 version_info")
                continue

            version_instance = version_class()
            get_timeversion = getattr(version_instance, "get_timeversion", None)
            if get_timeversion is None:
                print(f"{module_name}.version_info 中没有 get_timeversion")
                continue

            results[module_name] = get_timeversion()

        except Exception as e:
            print(f"{module_name} 处理失败: {e}")

    return results

print(get_all_timeversions())