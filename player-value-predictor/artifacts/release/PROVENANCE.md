# Serving bundle provenance

This is a byte-for-byte copy of the locally frozen Phase 2b artifacts.
No model refitting, metric changes or test-period scoring occurred during packaging.

Dataset: dcaribou/transfermarkt-datasets, recorded database revision
`e44f186d6f06dd8452aaf54c7921ba66c961f637`.

The upstream repository at that revision publishes the downloadable DuckDB
dataset and a CC0 1.0 Universal licence:

- https://github.com/dcaribou/transfermarkt-datasets/tree/e44f186d6f06dd8452aaf54c7921ba66c961f637
- https://github.com/dcaribou/transfermarkt-datasets/blob/e44f186d6f06dd8452aaf54c7921ba66c961f637/LICENSE

The licence is copied under `licenses/transfermarkt-datasets-CC0.txt`.
The public dataset originated from Transfermarkt. We claim no affiliation or
endorsement. CC0 includes third-party-rights and no-warranty limitations.

The release includes the fitted model, historical eligible player lookup,
small baseline history used by the frozen artifact integrity contract, feature
schema, selected-model metadata and already-recorded metrics. It excludes the
raw DuckDB database and the full training table. The API returns player age,
never date of birth. Lookup records are historical, not current squads.

`selection.json` preserves the original model/artifact/source hashes.
`release.json` additionally checksums every distributed file in this folder.
