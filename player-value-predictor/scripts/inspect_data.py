"""Phase 0: read-only look at the dataset's real structure. Changes nothing."""
import os
import sys
import duckdb

DEFAULT = "/mnt/data/datasets/transfermarkt/transfermarkt-datasets.duckdb"
db_path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("TM_DB", DEFAULT)

if not os.path.exists(db_path):
    sys.exit(f"Database file not found: {db_path}")

con = duckdb.connect(db_path, read_only=True)

tables = sorted(r[0] for r in con.execute("SHOW TABLES").fetchall())
print(f"File: {db_path}")
print(f"Size: {os.path.getsize(db_path) / 1e6:,.1f} MB")
print(f"\n{len(tables)} tables:")
for t in tables:
    n = con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
    print(f"  {t:22} {n:>12,} rows")

focus = ["players", "player_valuations", "appearances", "clubs", "competitions"]
for t in focus:
    print(f"\n--- columns of {t} ---")
    if t not in tables:
        print("  (table not found)")
        continue
    cols = con.execute(
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_name = ? ORDER BY ordinal_position", [t]
    ).fetchall()
    for name, dtype in cols:
        print(f"  {name:40} {dtype}")

for t in ["player_valuations", "appearances", "clubs"]:
    if t in tables:
        print(f"\n--- first 3 rows of {t} ---")
        print(con.execute(f'SELECT * FROM "{t}" LIMIT 3').df().to_string())
