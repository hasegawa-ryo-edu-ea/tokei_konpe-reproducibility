#!/usr/bin/env python3
"""Small immutable e-Stat API fetcher for explicitly selected tables/items."""
from __future__ import annotations
import argparse, datetime as dt, gzip, hashlib, json, time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE = "https://api.e-stat.go.jp/rest/3.0/app/json"

def api_key(env_path: Path) -> str:
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("ESTAT_APP_ID="):
            return line.split("=", 1)[1].strip().strip('"\'')
    raise RuntimeError("ESTAT_APP_ID not found")

def request(endpoint: str, params: dict[str, str], key: str) -> dict:
    query = urlencode({"appId": key, **params})
    last = None
    for attempt in range(5):
        try:
            with urlopen(Request(f"{BASE}/{endpoint}?{query}", headers={"User-Agent":"tokei-konpe-research/1.0"}), timeout=120) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as exc:
            last = exc; time.sleep(2 ** attempt)
    raise RuntimeError(f"e-Stat request failed: {endpoint}") from last

def iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True); p.add_argument("--out", required=True)
    p.add_argument("--table", required=True); p.add_argument("--items", default="")
    # Keep filters explicit in the immutable request manifest.  These are useful
    # for compact historical Census extracts whose full tables are very large.
    p.add_argument("--cat02", default=""); p.add_argument("--cat03", default="")
    p.add_argument("--metadata-only", action="store_true")
    a = p.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); key = api_key(Path(a.env))
    meta = request("getMetaInfo", {"statsDataId": a.table}, key)
    (out / f"{a.table}_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if a.metadata_only: return
    filters = {"statsDataId": a.table, "metaGetFlg":"N", "limit":"100000"}
    if a.items: filters["cdCat01"] = a.items
    if a.cat02: filters["cdCat02"] = a.cat02
    if a.cat03: filters["cdCat03"] = a.cat03
    digest = hashlib.sha256(json.dumps(filters, sort_keys=True).encode()).hexdigest()[:12]
    values = out / f"{a.table}_{digest}.jsonl.gz"; start = 1; rows = 0; pages = 0
    with gzip.open(values, "wt", encoding="utf-8") as fh:
        while True:
            r = request("getStatsData", {**filters, "startPosition":str(start)}, key)
            body = r["GET_STATS_DATA"]; result = body["RESULT"]
            if str(result["STATUS"]) != "0": raise RuntimeError(result)
            data = body["STATISTICAL_DATA"]; vals = data.get("DATA_INF",{}).get("VALUE",[])
            if isinstance(vals, dict): vals = [vals]
            for v in vals: fh.write(json.dumps(v, ensure_ascii=False)+"\n"); rows += 1
            pages += 1; nxt = data.get("RESULT_INF",{}).get("NEXT_KEY")
            if not nxt: break
            start = int(nxt)
    manifest = {"source":"e-Stat API getStatsData","retrieved_at":iso(),"stats_data_id":a.table,"filters":filters,"rows":rows,"pages":pages,"values_file":str(values)}
    (out / f"{a.table}_{digest}_request.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__": main()
