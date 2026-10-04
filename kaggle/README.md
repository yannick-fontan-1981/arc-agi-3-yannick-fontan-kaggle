# V5 Kaggle runtime bundle

This area generates the V5 runtime, competition supervisor and a portable notebook.
It does not upload or submit to Kaggle.
The authoritative agent remains in `agents/yf_arc3_v5/`. `submission/` is generated
and ignored by Git; rebuild after every source/resource change.

```sh
python kaggle/build_submission.py
python kaggle/build_notebook.py
python -m pytest kaggle/tests -q
```

Use an interpreter with Pydantic 2 and requests installed. On this workstation
the existing diagnostic environment can run these commands:

```powershell
api/.venv/Scripts/python.exe kaggle/build_submission.py
api/.venv/Scripts/python.exe kaggle/build_notebook.py
api/.venv/Scripts/python.exe -m pytest kaggle/tests -q
```

The target remains Python >=3.12, as declared by the project and official starter.
Local validation currently uses Python 3.10.4; this is useful packaging evidence,
but does not attest execution in the actual Python 3.12 Kaggle image. No packages
are installed or downloaded by this builder.

The bundle contains `my_agent.py`, shared lifecycle/action types, the recursive
V5 Python closure and referenced generic SRC/DRM/JSON declarations. Runtime Logos
Python is included. The unreferenced `core.logos` text and frozen V0.9 sources are
excluded. Paths and original source bytes are preserved. Only the generated
`agents/__init__.py` is inert, avoiding development template discovery and dotenv.
`adapter.py` is the maintained small source for generated `my_agent.py`.
The generated bundle also carries this directory's `LICENSE` and
`THIRD_PARTY_NOTICES.md` so your MIT terms and the retained ARC Prize attribution
travel with the ZIP.

Excluded: local API/engines, custom games, environment definitions, oracles, tests,
tools, docs, runs, UI, MCP servers, Codex configuration and LLM integrations. Shared
import dependencies such as Recorder stay in source, but recording is disabled.
The human-only recorder is lazy in V5 and its implementation is not shipped.

`MANIFEST.json` reports source HEAD, whether Git sees runtime source changes,
per-file SHA-256 and sizes, total files/bytes (including the manifest), dependencies
and builder version. Its own hash is excluded to avoid circular hashing. There is
no wall-clock timestamp. Identical source bytes, builder and Git identity generate
identical files. A failed build removes the candidate output.

`dependency_inventory.json` records imports, resources, execution-sensitive calls
and reviewed exclusions. `leakage_findings.json` preserves every suspicious reference;
`leakage_review.json` records exact reviewed signatures and reasons. A new finding
fails the build, removes the candidate and requires review. Do not bulk-approve
findings when updating V5. See `leakage_audit.md` and the dependency audit in docs.

The isolated smoke process uses Python `-I`, a copied bundle outside the repository,
cleared ARC/Python/telemetry variables, no `.env`, blocked sockets, blocked
subprocesses, and an audit hook rejecting development-repository source reads.
Installed third-party packages remain accessible, including this workstation's
venv site-packages. A real controller compiles resources, bootstraps from a tiny
synthetic observation, selects and dispatches through its authorization boundary,
and reconciles a terminal frame. Separate representation checks cover ACTION6
coordinates, ACTION7, HTTP serialization and modern `levels_completed` responses.
The mock point check does not prove cognitive point/undo selection. Tests also
cover session separation, rejected repeated bootstrap, terminal recovery without
a fallback RESET, exclusions, new leakage, hashes and deterministic rebuilds.
The compact result is written to `isolation_report.json`.

The official [starter notebook builder](https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter/blob/main/scripts/build_notebook.py)
uses an internal competition gateway. The local duck harness independently uses
that gateway during competition reruns and bundled official environments during
ordinary notebook runs. This bundle uses V5's ARC_API HTTP profile; inference
needs the permitted gateway, with no external Internet, LLM, GitHub or MCP service.
The adapter accepts `http://gateway:8001`, disables manual mode/recording and adds
no action policy. Supply the notebook's API key and scorecard/session orchestration
there. The existing V5 controller owns all releases and retained memory.

The current upstream Agent accepts `arc_env` and uses `levels_completed`; this
repository's shared base class has an older HTTP lifecycle. The adapter deliberately
rejects an `arc_env` argument until that exact mounted interface is integrated.
Do not overlay the upstream `agents` package onto this bundle or drop `MyAgent`
blindly into the current upstream runner. Official observation decoding already
normalizes `levels_completed` to V5 score. ACTION7 is now represented by the shared
enum; its use is governed by available actions and V5 authority. ACTION6 uses
top-level x/y in the REST command, per the
[official endpoint](https://docs.arcprize.org/api-reference/commands/execute-complex-action-requires-xy).

`build_notebook.py` creates `artifacts/arc3_v5_submission.ipynb` and the matching
`artifacts/v5_runtime.zip`. Attach the ZIP contents as a private Kaggle dataset,
import the notebook and attach the competition input. Both an intact ZIP and an
already extracted dataset are accepted after exact hash verification. Internet
stays disabled. Dependency installation uses only attached offline wheels.

The notebook runs at most **two game processes concurrently**, one continuous
controller per game. The user limits are **1,000 physical commands and 900 seconds
(15 minutes) per game**. RESET and every primitive of a series count. Limits may
be reduced, but the runner rejects increases beyond these ceilings. Time includes
launch/import/startup and all levels together. A fair share of the remaining
global budget may shorten a game's time; the notebook reserves 30 minutes from
its nine-hour allowance for closing/output. Neither budget is per level.

The runner calls V5's native `next_action()` cycle; it preserves declared series,
reconciliation and cognitive memory across levels. It bypasses the template's
80-iteration loop. V5 retains authority over actions and any lawful recovery.
The supervisor terminates a worker after five minutes without a command/report,
after its time budget or on a crash; remaining games continue. It does not restart
the game or replay an uncertain HTTP command. HTTP requests have a 30-second
timeout shortened to the remaining game time. Compact reports preserve attempted
and completed counts, last command, last state/score and uncertain delivery.

Save & Run All checks dependencies, hashes and isolated import, then writes the
official starter's placeholder `submission.parquet`. It does **not** measure
game performance. Competition rerun discovers IDs through the gateway, opens a
competition scorecard, runs the games and closes the card in `finally`. The
gateway must produce the actual `submission.parquet`; its absence fails the run.
No game engine or prerecorded solution is embedded in either artifact.

The package checks and isolated smoke test validate packaging and interface
behavior. A Kaggle score is a separate evaluation result and may vary between
submissions; follow the competition page for the current result.
