# Runtime requirements

## REQUIRED

- Python **>=3.12**. Root requirement is unchanged. The official
  [starter](https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter) specifies Python
  3.12 for the ARC package. Local tests ran on 3.10.4; actual Kaggle interpreter
  compatibility must still be executed in the notebook.
- `pydantic>=2.11.7,<3`: models, validators, immutable cognitive types. Includes
  pydantic-core, annotated-types, typing-extensions and typing-inspection.
- `requests>=2.32.4,<3`: shared Agent session construction, even when the injected
  transport itself uses stdlib urllib. Includes urllib3, certifi, idna and
  charset-normalizer. The local diagnostic venv warns that charset-normalizer /
  chardet is absent; install a complete requests wheel set in the notebook.

No dependency wheels are vendored. The builder performs no pip operation. Resolve
these and their transitive wheels offline from the competition wheelhouse or an
explicit notebook dataset; unavailable wheels are a notebook integration blocker.

## NOT REQUIRED BY V5 KAGGLE RUNTIME

Root dotenv, langchain[openai], langgraph, langgraph-checkpoint-sqlite, langsmith,
numpy, openai, pillow and smolagents are not in this agent's executed import closure.
There is no external LLM inference. Root `agents/__init__.py` is replaced only in
generated output to avoid pulling those templates and dotenv into inference.
AgentOps is optional and is not initialized. Pytest is a build/test dependency,
not a submitted inference dependency. ARC engine / arc-agi SDK are not required
by V5's direct HTTP transport or the new stdlib competition supervisor.
The notebook additionally checks `pandas` and `pyarrow` for writing the official
Save & Run All placeholder; these are not V5 cognition dependencies. Actual
competition output is written by the gateway.

## UNCERTAIN

- Exact Kaggle image version and offline wheel versions/availability.
- Actual gateway REST behavior and scorecard/output lifecycle. The generated
  notebook uses direct HTTP instead of the mounted upstream `arc_env` runner;
  no SDK wrapper compatibility is claimed.

Evidence: recursive AST inventory, isolated import and real synthetic controller
execution with no repository source access or network. No Python requirement
was lowered and no complete root dependency set was included.
