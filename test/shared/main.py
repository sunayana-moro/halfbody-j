"""Run every shared/ test suite in one shot.

    test_env/bin/python test/shared/main.py

Loads the sibling test files (layers.py, attention.py, stylegan.py) by path and
calls each one's main() — each runs its smoke() + parity(). Prints a per-suite
banner and a final aggregate. Exit 0 iff all suites pass.

Loading by path (not `import`) avoids name-shadowing the real torch/jax modules,
and loading with a non-"__main__" module name means each file's
`if __name__ == "__main__"` guard does NOT auto-run — we call mod.main() ourselves.
Do NOT import jax here: let the first loaded suite enable x64 before any array.
"""

import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SUITES = ("layers", "attention", "stylegan")


def load(name):
    path = os.path.join(HERE, name + ".py")
    spec = importlib.util.spec_from_file_location(f"suite_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    results = {}
    for name in SUITES:
        print("\n" + "#" * 84)
        print(f"#  SUITE: {name}")
        print("#" * 84 + "\n")
        results[name] = load(name).main()          # 0 = pass, 1 = fail

    n_pass = sum(1 for rc in results.values() if rc == 0)
    print("\n" + "#" * 84)
    print(f"#  OVERALL: {n_pass}/{len(SUITES)} SUITES PASS")
    for name, rc in results.items():
        print(f"#    {name:12s} {'PASS' if rc == 0 else 'FAIL'}")
    print("#" * 84)
    return 0 if n_pass == len(SUITES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
