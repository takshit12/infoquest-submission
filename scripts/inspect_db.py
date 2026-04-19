"""Read-only database inspection: list tables, schemas, row counts, sample rows."""
import os
import json
import psycopg2
import psycopg2.extras

DSN = os.environ.get("DATABASE_URL")
if not DSN:
    raise SystemExit("DATABASE_URL not set — populate .env from .env.example first")

def main():
    conn = psycopg2.connect(DSN, connect_timeout=15)
    conn.set_session(readonly=True, autocommit=True)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    print("=" * 80)
    print("SERVER VERSION")
    print("=" * 80)
    cur.execute("SELECT version()")
    print(cur.fetchone()["version"])

    print("\n" + "=" * 80)
    print("SCHEMAS (non-system)")
    print("=" * 80)
    cur.execute(
        """
        SELECT schema_name
        FROM information_schema.schemata
        WHERE schema_name NOT IN ('pg_catalog', 'information_schema', 'pg_toast')
          AND schema_name NOT LIKE 'pg_%'
        ORDER BY schema_name
        """
    )
    for r in cur.fetchall():
        print(" -", r["schema_name"])

    print("\n" + "=" * 80)
    print("TABLES (with row counts) in user schemas")
    print("=" * 80)
    cur.execute(
        """
        SELECT table_schema, table_name, table_type
        FROM information_schema.tables
        WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
          AND table_schema NOT LIKE 'pg_%'
        ORDER BY table_schema, table_name
        """
    )
    tables = cur.fetchall()
    for t in tables:
        qualified = f'"{t["table_schema"]}"."{t["table_name"]}"'
        try:
            cur.execute(f"SELECT COUNT(*) AS n FROM {qualified}")  # nosec B608 - identifier from pg_catalog query above, not user input
            n = cur.fetchone()["n"]
        except Exception as e:
            n = f"err: {e}"
        print(f" - {t['table_schema']}.{t['table_name']}  ({t['table_type']})  rows={n}")

    print("\n" + "=" * 80)
    print("COLUMN DEFINITIONS per table")
    print("=" * 80)
    for t in tables:
        if t["table_type"] != "BASE TABLE":
            continue
        print(f"\n-- {t['table_schema']}.{t['table_name']} --")
        cur.execute(
            """
            SELECT column_name, data_type, is_nullable, character_maximum_length
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            ORDER BY ordinal_position
            """,
            (t["table_schema"], t["table_name"]),
        )
        for c in cur.fetchall():
            mx = f"({c['character_maximum_length']})" if c["character_maximum_length"] else ""
            print(f"  {c['column_name']:<40} {c['data_type']}{mx:<6}  null={c['is_nullable']}")

    print("\n" + "=" * 80)
    print("INDEXES")
    print("=" * 80)
    cur.execute(
        """
        SELECT schemaname, tablename, indexname, indexdef
        FROM pg_indexes
        WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
          AND schemaname NOT LIKE 'pg_%'
        ORDER BY schemaname, tablename, indexname
        """
    )
    for r in cur.fetchall():
        print(f"  {r['schemaname']}.{r['tablename']}  {r['indexname']}")
        print(f"    {r['indexdef']}")

    print("\n" + "=" * 80)
    print("INSTALLED EXTENSIONS (look for pgvector)")
    print("=" * 80)
    cur.execute("SELECT extname, extversion FROM pg_extension ORDER BY extname")
    for r in cur.fetchall():
        print(f"  {r['extname']}  v{r['extversion']}")

    conn.close()

if __name__ == "__main__":
    main()
