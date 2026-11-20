#!/usr/bin/env python3
"""
gui_server.py -- the local web page for the exemplar generator.  Python standard library only (no extra packages); the page is plain HTML in gui_static/.

  python gui_server.py              starts the server on http://127.0.0.1:8765 and opens it in the browser
  python gui_server.py --demo       the same page with stand-in workers and made-up results (nothing real is computed; needs no model, Vina or RDKit)
  python gui_server.py --check      does NOT start the server: prints a pass/fail list of everything the page needs on this machine (files, packages, checkpoints, receptor files, the commands it would run)
  python gui_server.py --port 9000 --no-browser

The server listens on this computer only (127.0.0.1).  Every change request must carry the header X-GUI: 1, which a web page from another site cannot add.
All work is done by gui_backend.py, which starts the existing scripts as separate processes.
"""
import argparse
import json
import mimetypes
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

import config as cfg
import gui_backend as gb

STATIC_DIR = os.path.join(cfg.HERE, 'gui_static')
DEFAULT_PORT = 8765
MAX_BODY = 2_000_000


# Handles one HTTP request: static files for the page, JSON for /api/...
class Handler(BaseHTTPRequestHandler):
    server_version = 'ExemplarGUI/1.0'

    # Quiet log: only errors reach the console.
    def log_message(self, fmt, *args):
        if args and str(args[1]).startswith(('4', '5')):
            sys.stderr.write('[gui] ' + (fmt % args) + '\n')

    # Sends a JSON reply.
    def reply(self, obj, code=200):
        body = json.dumps(obj).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    # Sends a reply with a given content type.
    def send_bytes(self, data, ctype, code=200, extra=None):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(data)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    # Serves a file from gui_static/ (never outside it).
    def static(self, rel):
        rel = rel.lstrip('/') or 'index.html'
        if rel == 'favicon.ico':                      # browsers ask for it; there is none, so answer quietly instead of logging a 404
            self.send_response(204)
            self.end_headers()
            return
        path = os.path.normpath(os.path.join(STATIC_DIR, rel))
        if not path.startswith(STATIC_DIR + os.sep) or not os.path.isfile(path):
            return self.reply({'error': 'not found'}, 404)
        ctype = mimetypes.guess_type(path)[0] or 'application/octet-stream'
        with open(path, 'rb') as fh:
            self.send_bytes(fh.read(), ctype + ('; charset=utf-8' if ctype.startswith('text/') or ctype.endswith('javascript') else ''), extra={'Cache-Control': 'no-store'})

    # GET: the page, its files and the read-only API calls.
    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path.startswith('/api/'):
                return self.api_get(u.path[5:], q)
            return self.static(unquote(u.path))
        except ValueError as e:
            return self.reply({'error': str(e)}, 400)
        except Exception as e:
            return self.reply({'error': f'{type(e).__name__}: {e}'}, 500)

    # The read-only API: machine state, job status, runs, candidates, molecule pictures, 3D coordinates, CSV download.
    def api_get(self, path, q):
        if path == 'state':
            st = gb.detect_state()
            job = gb.current_job()
            st['job'] = job.snapshot(0) if job is not None else None
            return self.reply(st)
        if path == 'job':
            job = gb.current_job()
            if job is None:
                return self.reply({'job': None})
            return self.reply({'job': job.snapshot(int(q.get('since', 0)))})
        if path == 'runs':
            return self.reply({'runs': gb.list_runs()})
        if path == 'candidates':
            return self.reply(gb.candidates(q.get('run', ''), int(q.get('limit', 60))))
        if path == 'svg':
            return self.send_bytes(gb.svg(q.get('smiles', '')).encode('utf-8'), 'image/svg+xml')
        if path == 'mol3d':
            return self.reply(gb.molecule_3d(q.get('smiles', '')))
        if path == 'download':
            run = q.get('run', '')
            name = {'gen': 'gen.csv', 'dock': 'dock_results.csv', 'log': 'log.txt'}.get(q.get('file', ''))
            p = os.path.join(gb.RUNS_DIR, run, name or '')
            if not name or not run.replace('-', '').replace('_', '').isalnum() or not os.path.isfile(p):
                return self.reply({'error': 'not found'}, 404)
            with open(p, 'rb') as fh:
                return self.send_bytes(fh.read(), 'text/csv; charset=utf-8' if name.endswith('csv') else 'text/plain; charset=utf-8',
                                       extra={'Content-Disposition': f'attachment; filename="{run}_{name}"'})
        return self.reply({'error': 'unknown call'}, 404)

    # POST: start a job, cancel it, save a pasted ligand library.
    def do_POST(self):
        if self.headers.get('X-GUI') != '1':
            return self.reply({'error': 'missing X-GUI header'}, 403)
        try:
            n = int(self.headers.get('Content-Length', 0))
            if n > MAX_BODY:
                return self.reply({'error': 'request too large'}, 413)
            body = json.loads(self.rfile.read(n) or b'{}')
            path = urlparse(self.path).path
            if path == '/api/start':
                return self.reply({'job': gb.start_job(body).snapshot(0)})
            if path == '/api/cancel':
                job = gb.current_job()
                if job is not None:
                    job.cancel()
                return self.reply({'ok': True})
            if path == '/api/library':
                return self.reply(gb.save_library(body.get('disease'), body.get('target'), body.get('smiles')))
            return self.reply({'error': 'unknown call'}, 404)
        except ValueError as e:
            return self.reply({'error': str(e)}, 400)
        except RuntimeError as e:
            return self.reply({'error': str(e)}, 409)
        except Exception as e:
            return self.reply({'error': f'{type(e).__name__}: {e}'}, 500)


# Self-check for a new machine: prints one line per requirement (OK / MISSING / FAIL) and returns the number of problems.  Nothing is started and nothing is written.
def check(port):
    import importlib.util
    import socket
    problems = []

    def line(ok, what, detail='', warn=False):
        tag = 'OK     ' if ok else ('WARN   ' if warn else 'MISSING')
        print(f"  [{tag}] {what}" + (f"   {detail}" if detail else ''))
        if not ok and not warn:
            problems.append(what)

    print(f"GUI self-check   (python {sys.version.split()[0]}, folder {cfg.HERE})")
    print("\nthe page")
    for f in ('index.html', 'style.css', 'app.js'):
        line(os.path.isfile(os.path.join(STATIC_DIR, f)), f'gui_static/{f}', 'the gui_static folder must sit next to gui_server.py')
    print("\nscripts the page starts")
    for f in ('config.py', 'targets.py', 'generate_for_request.py', 'dock_generated.py', 'check_exemplar_conditioning.py', 'train_exemplar_generator.py', 'exemplar_model.py', 'exemplar_data.py', 'docking.py'):
        line(os.path.isfile(os.path.join(cfg.HERE, f)), f)
    print("\npython packages (not installed by this check)")
    for mod in ('numpy', 'pandas', 'torch', 'rdkit', 'meeko'):
        line(importlib.util.find_spec(mod) is not None, mod)
    print("\ndata, model and docking files")
    st = gb.detect_state()
    line(st['library'], cfg.CSV_NAME)
    line(st['vina'], 'vina.exe', cfg.DEFAULT_VINA)
    ex = [c for c in st['checkpoints'] if c['role'] == 'exemplar']
    line(bool(ex), 'exemplar checkpoint(s)', ', '.join(c['name'] for c in ex) or 'none found')
    line(any(c['role'] == 'pretrained' for c in st['checkpoints']), cfg.PRETRAINED_CKPT_NAME, 'only needed for the Train stage', warn=True)
    for t in st['targets']:
        if t.get('dock'):
            for r in t['requirements']:
                line(r['found'], f"{t['label']}: {r['file']}")
    print("\nthe commands the page would run (nothing is started)")
    for t in gb.STUDY_TARGETS:
        stages = ['generate', 'baseline', 'dock'] if t.get('dock') else ['generate']
        try:
            run_id, run_dir, params, plan = gb.plan_pipeline({'target_key': t['key'], 'stages': stages, 'n': 10, 'checkpoint': st['default_checkpoint']})
            line(True, f"{t['label']}: {' + '.join(stages)}")
            for p_ in plan:
                print('           ' + ' '.join(os.path.basename(x) if i < 3 else x for i, x in enumerate(p_['argv'])).replace(cfg.HERE, '.')[:700])
        except Exception as e:
            line(False, f"{t['label']}: could not build the commands", f"{type(e).__name__}: {e}")
    print("\nmolecule pictures and properties (need RDKit)")
    for name, fn in (('2D picture', lambda: gb.svg('c1ccccc1')), ('properties', lambda: gb.properties('c1ccccc1O')), ('3D coordinates', lambda: gb.molecule_3d('CCO'))):
        try:
            r = fn()
            ok = bool(r) and not (isinstance(r, dict) and r.get('error')) and 'RDKit not installed' not in str(r) and 'invalid SMILES' not in str(r)
            line(ok, name, '' if ok else 'RDKit is missing or could not draw this molecule', warn=True)
        except Exception as e:
            line(False, name, f"{type(e).__name__}: {str(e)[:80]}", warn=True)
    print("\nport")
    s_ = socket.socket()
    try:
        s_.bind(('127.0.0.1', port))
        line(True, f'127.0.0.1:{port} is free')
    except OSError:
        line(False, f'127.0.0.1:{port} is in use', 'close the other server or use --port 9000', warn=True)
    finally:
        s_.close()
    print(f"\n{'ALL REQUIRED ITEMS PRESENT' if not problems else str(len(problems)) + ' PROBLEM(S): ' + '; '.join(problems)}")
    return len(problems)


# Reads the options, starts the server and (unless --no-browser) opens the page.
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port', type=int, default=DEFAULT_PORT)
    ap.add_argument('--demo', action='store_true', help='stand-in workers and made-up results')
    ap.add_argument('--no-browser', action='store_true')
    ap.add_argument('--check', action='store_true', help='print a pass/fail list of what the page needs on this machine and exit')
    a = ap.parse_args()
    gb.DEMO = a.demo
    if a.check:
        sys.exit(1 if check(a.port) else 0)
    os.makedirs(gb.RUNS_DIR, exist_ok=True)
    srv = ThreadingHTTPServer(('127.0.0.1', a.port), Handler)
    url = f'http://127.0.0.1:{a.port}/'
    print(f"Exemplar generator GUI{' (DEMO MODE: made-up results)' if a.demo else ''} at {url}   (Ctrl+C to stop)")
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        job = gb.current_job()
        if job is not None and job.status == 'running':
            job.cancel()
        print('stopped')


if __name__ == '__main__':
    main()
