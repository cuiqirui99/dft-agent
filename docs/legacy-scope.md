# Relationship to the earlier platform

This release packages the basic VASP/Slurm workflow as a standalone application.
It reuses the earlier project's deterministic workflow approach but does not
import its multimodal platform, private datasets or historical experiment state.

The 2026-10-07 audit of the earlier workspace recorded **911 passed, 93 failed
and 2 errors** out of 1,006 tests. These results have not been relabelled as a
passing suite. The [classification receipt](../validation/legacy-scope.json)
identifies the source report by SHA-256:

| Unresolved category | Count | Treatment in this release |
| --- | ---: | --- |
| Missing historical evidence or dataset files | 89 | Preserve the old tests and records; do not reconstruct datasets to ship the basic tool |
| Frozen dependency hash contracts in advanced DFT campaigns | 5 | Keep the defect/surface and k-point research campaigns outside the basic recipes |
| Pinned XPS parser interpreter contract | 1 | XPS is outside the release scope |

All affected legacy test files remain present. The new package's passing test
count describes only this package. Its VASP calculations have new input/output
receipts and real cluster acceptance; they do not inherit a scientific pass
from an older report. The SSH transfer, job identity, result checking and
installation behavior are verified against the new package directly.
