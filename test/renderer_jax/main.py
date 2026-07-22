"""Run all renderer_jax test suites.

    test_env/bin/python test/renderer_jax/main.py

Loads each sibling suite by path and calls its main() (smoke + parity). Do NOT
import jax here — let the first suite enable x64 before any array is created.
"""

import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SUITES = ("encoders", "decoders", "renderer_smoke")


def load(name):
    spec = importlib.util.spec_from_file_location(f"rj_suite_{name}", os.path.join(HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    results = {}
    for name in SUITES:
        print("\n" + "#" * 84)
        print(f"#  SUITE: {name}")
        print("#" * 84 + "\n")
        results[name] = load(name).main()

    n_pass = sum(1 for rc in results.values() if rc == 0)
    print("\n" + "#" * 84)
    print(f"#  OVERALL: {n_pass}/{len(SUITES)} SUITES PASS")
    for name, rc in results.items():
        print(f"#    {name:12s} {'PASS' if rc == 0 else 'FAIL'}")
    print("#" * 84)
    return 0 if n_pass == len(SUITES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
