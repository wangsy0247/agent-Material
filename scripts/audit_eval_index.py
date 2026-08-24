#!/usr/bin/env python3
"""审计 PGVector 评测索引的版本化元数据和控制面状态。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import psycopg
from psycopg import sql

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "packages"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from agent_material.rag_engine.db.pg.schema import normalize_conn_str  # noqa: E402
from agent_material.utils.config_handler import vector_store_config  # noqa: E402


def _collection_stats(conn, name: str, *, child: bool) -> dict:
    required = ["document_id", "version_id", "parent_id", "is_active"]
    if child:
        required.append("chunk_id")
    predicates = sql.SQL(" AND ").join(
        sql.SQL("e.cmetadata ? {}").format(sql.Literal(key)) for key in required
    )
    query = sql.SQL(
        """
        SELECT
            COUNT(e.id) AS total_rows,
            COUNT(e.id) FILTER (WHERE e.cmetadata->>'is_active' = 'true') AS active_rows,
            COUNT(e.id) FILTER (WHERE NOT ({required})) AS missing_required_rows,
            COUNT(DISTINCT e.cmetadata->>'document_id')
                FILTER (WHERE e.cmetadata->>'is_active' = 'true') AS active_documents,
            COUNT(DISTINCT e.cmetadata->>'version_id')
                FILTER (WHERE e.cmetadata->>'is_active' = 'true') AS active_versions
        FROM langchain_pg_collection c
        LEFT JOIN langchain_pg_embedding e ON e.collection_id = c.uuid
        WHERE c.name = %s
        """
    ).format(required=predicates)
    row = conn.execute(query, (name,)).fetchone()
    return {
        "collection": name,
        "total_rows": row[0],
        "active_rows": row[1],
        "missing_required_rows": row[2],
        "active_documents": row[3],
        "active_versions": row[4],
        "required_metadata": required,
    }


def audit(prefix: str, kb_id: str) -> dict:
    conn_str = normalize_conn_str(vector_store_config["connection_string"])
    with psycopg.connect(conn_str) as conn:
        collections = [
            _collection_stats(conn, f"{prefix}_small", child=True),
            _collection_stats(conn, f"{prefix}_large", child=False),
        ]
        sources = conn.execute(
            """
            SELECT c.name, COALESCE(e.cmetadata->>'source', '<missing>'), COUNT(*)
            FROM langchain_pg_collection c
            JOIN langchain_pg_embedding e ON e.collection_id = c.uuid
            WHERE c.name IN (%s, %s)
              AND e.cmetadata->>'is_active' = 'true'
            GROUP BY c.name, COALESCE(e.cmetadata->>'source', '<missing>')
            ORDER BY c.name, COUNT(*) DESC, 2
            """,
            (f"{prefix}_small", f"{prefix}_large"),
        ).fetchall()
        kb_state = conn.execute(
            """
            SELECT kb_id, active_revision, bm25_revision, updated_at
            FROM rag_kb_state WHERE kb_id = %s
            """,
            (kb_id,),
        ).fetchone()
        tasks = conn.execute(
            """
            SELECT status, COUNT(*)
            FROM rag_ingest_tasks
            WHERE payload->>'kb_id' = %s
            GROUP BY status ORDER BY status
            """,
            (kb_id,),
        ).fetchall()
        versions = conn.execute(
            """
            SELECT v.status, COUNT(*)
            FROM rag_document_versions v
            JOIN rag_documents d ON d.document_id = v.document_id
            WHERE d.kb_id = %s
            GROUP BY v.status ORDER BY v.status
            """,
            (kb_id,),
        ).fetchall()

    clean = all(
        c["total_rows"] > 0
        and c["active_rows"] > 0
        and c["missing_required_rows"] == 0
        and c["total_rows"] == c["active_rows"]
        and c["active_documents"] == 1
        and c["active_versions"] == 1
        for c in collections
    ) and kb_state is not None
    return {
        "kb_id": kb_id,
        "collection_prefix": prefix,
        "clean": clean,
        "collections": collections,
        "active_sources": [
            {"collection": name, "source": source, "rows": count}
            for name, source, count in sources
        ],
        "kb_state": (
            {
                "kb_id": kb_state[0],
                "active_revision": kb_state[1],
                "bm25_revision": kb_state[2],
                "updated_at": kb_state[3].isoformat(),
            }
            if kb_state
            else None
        ),
        "task_counts": dict(tasks),
        "version_counts": dict(versions),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collection-prefix",
        default=vector_store_config.get("character_name", "agent"),
    )
    parser.add_argument(
        "--kb-id", default=vector_store_config.get("character_name", "agent")
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = audit(args.collection_prefix, args.kb_id)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if result["clean"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
