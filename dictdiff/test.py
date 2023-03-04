from typing import Dict, Hashable, Union, Tuple, Any
from pprint import pprint
from copy import deepcopy
import unittest


class DictTuple(tuple):
    """Used for separating from regular tuples"""

    def __repr__(self):
        return f"#{super().__repr__()[1:-1]}#"


class _Undefined:
    """Flag for non-existing value"""

    def __repr__(self):
        return "undefined"


undefined = _Undefined()


def recursiveitems(d: Dict):
    for key, val in d.items():
        if type(val) == dict:
            yield (key, DictTuple(recursiveitems(val)))
        elif type(val) == list:
            yield (key, tuple(val))
        else:
            yield (key, val)


def difference(a: Dict, b: Dict):
    a_set = set(recursiveitems(a))
    b_set = set(recursiveitems(b))
    diffs = a_set.symmetric_difference(b_set)
    changes = {}
    for key, val in diffs:
        if type(val) == DictTuple:
            changes[key] = difference(a.get(key, undefined), b.get(key, undefined))
        else:
            changes[key] = (a.get(key, undefined), b.get(key, undefined))

    return changes


def _apply(
    d: Dict,
    changes: Dict[Hashable, Union[Tuple[Any, Any], Dict]],
    delete: bool,
    add: bool,
    update: bool,
):
    for key, val in changes.items():
        if type(val) == dict:
            _apply(d.get(key), val, delete, add, update)
            continue

        before, after = val
        if delete and (after is undefined):
            del d[key]
            continue

        if (add and (before is undefined)) or update:
            d[key] = after
            continue

    return d


def apply(
    d: Dict,
    diff: Dict[Hashable, Union[Tuple[Any, Any], Dict]],
    *,
    delete: bool = True,
    add: bool = True,
    update: bool = True,
    mutate: bool = False,
):
    target = d if mutate else deepcopy(d)
    result = _apply(target, diff, delete, add, update)
    return None if mutate else result


class Test(unittest.TestCase):
    def test_recursiveitems(self):
        d = {
            "a": 1,
            "b": "A string",
            "c": (1, 2, 3),
            "d": [4, 5, 6],
            "e": {
                "f": None,
            },
        }

        items = list(recursiveitems(d))
        expected = [
            ("a", 1),
            ("b", "A string"),
            ("c", (1, 2, 3)),
            ("d", (4, 5, 6)),
            (
                "e",
                DictTuple(
                    [
                        ("f", None),
                    ]
                ),
            ),
        ]

        self.assertListEqual(items, expected)

    a = {
        "a": 1,
        "b": "A string",
        "c": (1, 2, 3),
        "d": [1, 2, 3],
        "e": {"f": 1, "g": {}},
        "i": {"j": 1, "k": {"l": 0}},
        "m": 123,
        "n": frozenset((7, 6, 8)),
    }

    b = {
        "a": 1,
        "b": "A string!!!",
        "c": (1, 2, 3),
        "d": [2, 4],
        "e": {"f": 1, "g": {"h": "help"}},
        "i": {"j": 43, "k": {"l": 0}},
        "n": frozenset((9, 8)),
    }

    diff = {
        "b": ("A string", "A string!!!"),
        "d": ([1, 2, 3], [2, 4]),
        "e": {"g": {"h": (undefined, "help")}},
        "i": {"j": (1, 43)},
        "m": (123, undefined),
        "n": (frozenset((7, 6, 8)), frozenset((9, 8))),
    }

    def test_diff(self):
        diff = difference(self.a, self.b)
        self.assertDictEqual(diff, self.diff)

    def test_apply(self):
        b = apply(self.a, self.diff)
        self.assertDictEqual(b, self.b)

    def test_apply_mutate(self):
        a = deepcopy(self.a)
        apply(a, self.diff, mutate=True)
        self.assertDictEqual(a, self.b)

    def test_apply_update_only(self):
        a = {
            "a": {
                "delete": 1,
                "update": 3,
            },
        }

        diff = {
            "a": {
                "delete": (1, undefined),
                "update": (3, 2),
                "add": (undefined, 123),
            }
        }

        b = apply(a, diff)
        expected = {
            "a": {"update": 2, "add": 123},
        }

        self.assertDictEqual(b, expected)

    def test_apply_delete_only(self):
        a = {
            "a": {
                "delete": 1,
                "update": 3,
            },
        }

        diff = {
            "a": {
                "delete": (1, undefined),
                "update": (3, 2),
                "add": (undefined, 123),
            }
        }

        b = apply(a, diff, update=False, add=False)
        expected = {
            "a": {
                "update": 3,
            },
        }

        self.assertDictEqual(b, expected)

    def test_apply_add_only(self):
        a = {
            "a": {
                "delete": 1,
                "update": 3,
            },
        }

        diff = {
            "a": {
                "delete": (1, undefined),
                "update": (3, 2),
                "add": (undefined, 123),
            }
        }

        b = apply(a, diff, update=False, delete=False)
        expected = {
            "a": {
                "delete": 1,
                "update": 3,
                "add": 123,
            },
        }

        self.assertDictEqual(b, expected)


if __name__ == "__main__":
    unittest.main()

    # res = diff(a, b)
    # pprint(a, width=1)
    # apply(a, res, delete=True, add=True, update=True)
    # pprint(a, width=1)
