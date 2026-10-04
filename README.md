# ARC-AGI-3 — Yannick Fontan Kaggle package

This repository contains the source closure and notebook package for Yannick
Fontan's ARC-AGI-3 Kaggle agent. It is a curated release snapshot; it does not
contain the private development repository, custom games, game engines, oracle
solutions, or development history.

## Contents

- `agents/`: the Python source and declarative resources used by the agent.
- `kaggle/`: bundle and notebook builders, runtime documentation, focused
  packaging tests, and generated Kaggle artifacts.
- `kaggle/artifacts/v5_runtime.zip`: the runtime package to attach as a private
  Kaggle notebook input.
- `kaggle/artifacts/arc3_v5_submission.ipynb`: the notebook that validates and
  loads that package.
- `LICENSE` and `THIRD_PARTY_NOTICES.md`: the license for Yannick's work and
  preserved third-party attribution.

## Build and validate

Use Python 3.12, then install the local packaging/test dependencies:

```bash
python -m pip install "pydantic>=2.11.7,<3" "requests>=2.32.4,<3" pytest
python -m pytest kaggle/tests -q
python kaggle/build_notebook.py
```

The build creates `kaggle/submission/` and refreshes the notebook and ZIP under
`kaggle/artifacts/`. The submitted ZIP includes its manifest, agent source,
`LICENSE`, and `THIRD_PARTY_NOTICES.md`. Python libraries are not vendored in the
ZIP; the Kaggle notebook installs from its attached offline wheelhouse.

## Use on Kaggle

Import `kaggle/artifacts/arc3_v5_submission.ipynb` as a Kaggle notebook. Attach
`v5_runtime.zip` as a private notebook input and attach the ARC Prize 2026
competition input. Keep Internet disabled and use **Save & Run All** for the
package check. That check writes the official starter placeholder parquet; the
competition rerun uses the provided gateway to produce the scored output.

The runner uses at most two game workers concurrently and caps each game at
1,000 primitive commands or 900 seconds (15 minutes), including startup and all
levels. A stalled or crashed worker is recorded and the remaining games
continue. See [`kaggle/README.md`](kaggle/README.md) for the full runtime
contract and current validation limits.

## License and scope

Yannick Fontan's original work in this release is under the MIT License. The
shared ARC-AGI-3 agent lifecycle/data-model portions retain ARC Prize's MIT
copyright and permission notice in `THIRD_PARTY_NOTICES.md`. Runtime dependency
licenses are listed there; package wheels are installed separately and are not
redistributed in this repository's ZIP.

Local packaging tests and isolated controller smoke checks validate the
artifact structure and interface. They do not by themselves establish a score
or guarantee the behavior of a future Kaggle gateway run.
