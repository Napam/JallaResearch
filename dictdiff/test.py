from typing import Dict, Hashable, Union, Tuple, Any
from pprint import pprint


def items(d: Dict):
    for key, val in d.items():
        if isinstance(val, dict):
            yield (key, tuple(items(val)))
        else:
            yield (key, val)


def diff(a: Dict, b: Dict):
    a_set = set(items(a))
    b_set = set(items(b))
    diffs = a_set.symmetric_difference(b_set)
    # return {key: (a.get(key, None), b.get(key, None)) for key, val in diff}
    changes = {}
    for key, val in diffs:
        if isinstance(val, tuple):
            changes[key] = diff(a.get(key, None), b.get(key, None))
        else:
            changes[key] = (a.get(key, None), b.get(key, None))

    return changes


def diffupdate(
    d: Dict,
    changes: Dict[Hashable, Union[Tuple[Any, Any], Dict]],
    delete: bool = True,
    add: bool = True,
    update: bool = True,
) -> None:
    for key, val in changes.items():
        if isinstance(val, dict):
            diffupdate(d.get(key), val, delete, add, update)
        else:
            before, after = val
            del_ = after is None
            add_ = before is None

            if delete and del_:
                del d[key]
                continue

            if add and add_:
                d[key] = after
                continue

            if update and (not del_ and not add_):
                d[key] = after
                continue


if __name__ == "__main__":
    a = {
        "a": 1,
        "b": 2,
        "c": 3,
        "e": {
            "f": 1,
            "g": {
                "lol": "bananaa",
            },
        },
    }

    b = {
        "a": 60,
        "b": 3,
        "d": 4,
        "e": {
            "f": 1,
            "g": {
                "lol": "banana",
            },
        },
    }

    res = diff(a, b)

    pprint(a, width=1)
    diffupdate(a, res, delete=True, add=True, update=False)
    pprint(a, width=1)
