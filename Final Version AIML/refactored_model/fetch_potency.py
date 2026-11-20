#!/usr/bin/env python3
"""
fetch_potency.py -- download measured potencies (ChEMBL) for the three docking-study targets, for dock_generated.py --potency.

Run it ONCE on a machine with internet access (python 3, no extra packages).  It writes one CSV per case:
    potency_malaria_t89515.csv   (Plasmodium falciparum DHFR-TS, UniProt P13922)
    potency_diabetes.csv         (human DPP-4,                   UniProt P27487)
    potency_depression.csv       (human SERT,                    UniProt P31645)
Columns: smiles, chembl_id, type, pchembl, nM, assay_type, assay_id, target_chembl_id.
Only Ki / Kd / IC50 measurements with relation '=' and a ChEMBL pChEMBL value are kept (pChEMBL = -log10 of the molar value), so units are already
normalised; nM = 10 ** (9 - pChEMBL).  dock_generated.py takes the MEDIAN over a compound's measurements.

  python fetch_potency.py                      all three cases
  python fetch_potency.py --cases depression   one case
  python fetch_potency.py --types Ki,IC50      restrict the measurement types (default Ki,Kd,IC50)

The ChEMBL target found for each UniProt accession is printed: check that it is the species and protein you expect.
"""
import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

# ChEMBL web service.
HOST = 'https://www.ebi.ac.uk'
API = HOST + '/chembl/api/data/'
# Study case -> (UniProt accession of the protein, organism the ChEMBL target must belong to).  The accession is how the ChEMBL target is found.
CASES = {
    'malaria_t89515': ('P13922', 'Plasmodium falciparum'),
    'diabetes': ('P27487', 'Homo sapiens'),
    'depression': ('P31645', 'Homo sapiens'),
}
# Columns of every output CSV.
COLUMNS = ['smiles', 'chembl_id', 'type', 'pchembl', 'nM', 'assay_type', 'assay_id', 'target_chembl_id']


# GET a URL and return the parsed JSON.  Rate-limit and server errors (429, 5xx) and network timeouts are retried with a doubling delay (2 s, 4 s ...), up to `tries` attempts; anything else is raised.
def get_json(url, tries=6):
    delay = 2.0
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={'Accept': 'application/json', 'User-Agent': 'fetch_potency/1.0'})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and k < tries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            if k < tries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise


# Turns one page of the ChEMBL activity endpoint into rows (see COLUMNS).  Rows without a SMILES or a pChEMBL value, with a relation other than '=', or of a type not in keep_types are skipped.
# nM is computed from pChEMBL as 10 ** (9 - pChEMBL).
def parse_activities(page, keep_types):
    rows = []
    for a in page.get('activities', []):
        smi, pc = a.get('canonical_smiles'), a.get('pchembl_value')
        if not smi or pc in (None, ''):
            continue
        if a.get('standard_relation') not in (None, '='):
            continue
        t = a.get('standard_type')
        if t not in keep_types:
            continue
        try:
            pc = float(pc)
        except ValueError:
            continue
        rows.append({'smiles': smi, 'chembl_id': a.get('molecule_chembl_id', ''), 'type': t, 'pchembl': pc, 'nM': round(10 ** (9.0 - pc), 6),
                     'assay_type': a.get('assay_type', ''), 'assay_id': a.get('assay_chembl_id', ''), 'target_chembl_id': a.get('target_chembl_id', '')})
    return rows


# Looks up the ChEMBL targets that carry the UniProt accession, prints each one (so the species and protein can be checked) and returns the ids of the SINGLE PROTEIN targets of the expected organism.
def find_targets(accession, organism):
    url = API + 'target.json?' + urllib.parse.urlencode({'target_components__accession': accession, 'limit': 50})
    out = []
    for t in get_json(url).get('targets', []):
        print(f"      ChEMBL target {t.get('target_chembl_id')}: {t.get('pref_name')} | {t.get('organism')} | {t.get('target_type')}")
        if t.get('target_type') == 'SINGLE PROTEIN' and organism.lower() in str(t.get('organism', '')).lower():
            out.append(t['target_chembl_id'])
    return out


# Downloads all usable potency measurements for one study case and writes potency_<case>.csv.  For each matching ChEMBL target it follows the result pages (1000 rows each, at most max_pages) with a short pause between requests.
# Returns the file path, or None if no matching target exists.
def fetch_case(key, keep_types, out_dir, max_pages):
    acc, org = CASES[key]
    print(f"\n[{key}] UniProt {acc} ({org})")
    tids = find_targets(acc, org)
    if not tids:
        print("   no matching SINGLE PROTEIN target found: nothing written (check the accession / ChEMBL status)")
        return None
    rows = []
    for tid in tids:
        q = {'target_chembl_id': tid, 'pchembl_value__isnull': 'false', 'standard_type__in': ','.join(keep_types), 'limit': 1000}
        url = API + 'activity.json?' + urllib.parse.urlencode(q)
        page_no = 0
        while url and page_no < max_pages:
            page = get_json(url)
            rows += parse_activities(page, set(keep_types))
            nxt = (page.get('page_meta') or {}).get('next')
            url = (HOST + nxt) if nxt else None
            page_no += 1
            print(f"   {tid}: page {page_no}, {len(rows)} usable measurements so far", end='\r')
            time.sleep(0.3)
        print()
    path = os.path.join(out_dir, f"potency_{key}.csv")
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    n_comp = len({r['chembl_id'] for r in rows})
    print(f"   wrote {len(rows)} measurements for {n_comp} compounds -> {path}")
    return path


# Command-line entry point: runs fetch_case for each requested case; a failing case is reported and does not stop the others.  Exits with status 1 if any file was not written.
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cases', default=','.join(CASES), help='comma list of: ' + ', '.join(CASES))
    ap.add_argument('--types', default='Ki,Kd,IC50')
    ap.add_argument('--out-dir', default='.')
    ap.add_argument('--max-pages', type=int, default=200, help='safety cap on result pages (1000 rows each)')
    a = ap.parse_args()
    keep = [t.strip() for t in a.types.split(',') if t.strip()]
    todo = [c.strip() for c in a.cases.split(',') if c.strip()]
    bad = [c for c in todo if c not in CASES]
    if bad:
        raise SystemExit(f"unknown case(s) {bad}; choose from {list(CASES)}")
    done = []
    for c in todo:
        try:
            p = fetch_case(c, keep, a.out_dir, a.max_pages)
        except Exception as e:
            print(f"\n   FAILED: {type(e).__name__}: {e}")
            p = None
        if p:
            done.append(p)
    print(f"\n[done] {len(done)} of {len(todo)} files written")
    if len(done) < len(todo):
        sys.exit(1)


if __name__ == '__main__':
    main()
