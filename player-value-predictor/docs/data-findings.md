# Dataset provenance and publication check

Checked 7 November 2026 for Phase 5 packaging.

The local source database records revision
`e44f186d6f06dd8452aaf54c7921ba66c961f637` in its `version` table.
The matching upstream repository is
[dcaribou/transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets/tree/e44f186d6f06dd8452aaf54c7921ba66c961f637).
Its README at that revision links the downloadable DuckDB dataset, and its
[LICENSE](https://github.com/dcaribou/transfermarkt-datasets/blob/e44f186d6f06dd8452aaf54c7921ba66c961f637/LICENSE)
is CC0 1.0 Universal. The complete text is retained in
`licenses/transfermarkt-datasets-CC0.txt`.

This establishes the upstream publisher's stated licence at the recorded
revision. CC0 includes no-warranty and third-party-rights limitations; it does
not imply endorsement by Transfermarkt. The application is an educational
estimator of listed values, not transfer fees or players' true worth.

The release ships only the small frozen serving/evaluation artifacts and their
provenance. Raw data and the full training table remain excluded. Historical
lookup data contains age and not date of birth. No scraping is implemented.
The copied release preserves every original artifact byte and records its own
checksums. Existing final-test metrics are copied, never recomputed for release.

Earlier data checks are retained in `verify_output.txt` and
`verify_output_2.txt`. Valuations end on 2026-06-12; the model card documents
training, selection, calibration, final test results and limitations.
