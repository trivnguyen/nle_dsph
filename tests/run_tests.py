"""Run every `test_*` function in the test modules beside this file.

No framework: pytest is not a dependency of this project, and these
checks have to run anywhere the package does.

    python tests/run_tests.py            # all modules
    python tests/run_tests.py profiles   # one module, by suffix
"""

import importlib
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODULES = ('test_profiles', 'test_transforms', 'test_datasets',
           'test_observation')


def close(got, want, rel=1e-7, abs_=0.0):
    """True when `got` matches `want` to a relative or absolute tolerance."""
    return abs(got - want) <= max(abs_, rel * abs(want))


def raises(exc_type, match, func, *args, **kwargs):
    """True when `func` raises `exc_type` with `match` in its message."""
    try:
        func(*args, **kwargs)
    except exc_type as error:
        return match in str(error)
    return False


def main(selectors):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    names = [m for m in MODULES
             if not selectors or any(s in m for s in selectors)]

    passed, failed = 0, []
    for name in names:
        module = importlib.import_module(name)
        for attr in sorted(vars(module)):
            if not attr.startswith('test_'):
                continue
            try:
                getattr(module, attr)()
                passed += 1
            except Exception:
                failed.append(f'{name}.{attr}')
                print(f'FAIL {name}.{attr}')
                traceback.print_exc()

    print(f'\n{passed} passed, {len(failed)} failed')
    for name in failed:
        print(f'  failed: {name}')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
