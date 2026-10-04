# Third-party notices for the Kaggle runtime

The original adapter, supervisor and V5 agent material in this Kaggle package
is licensed under the MIT License in [`LICENSE`](LICENSE). Portions of the
shared ARC agent lifecycle and data model are derived from the ARC-AGI-3 Agents
project and remain subject to the following upstream notice:

```text
MIT License

Copyright (c) 2025 ARC Prize

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Python packages

The runtime ZIP does not vendor Python packages. Kaggle installs them from the
attached offline wheelhouse or uses the notebook environment. The direct
packages used by the runtime and notebook are:

| Package | License |
| --- | --- |
| Pydantic | MIT |
| Requests | Apache-2.0 |
| pandas | BSD-3-Clause |
| PyArrow | Apache-2.0 |

Their transitive dependencies are also installed packages, not copied into this
ZIP. Each installed distribution carries its own upstream license metadata and
notices. If package wheels are ever included in a release artifact, preserve
the notices from those exact wheel versions too.
