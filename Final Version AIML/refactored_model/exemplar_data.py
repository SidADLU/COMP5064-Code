#!/usr/bin/env python3
"""
exemplar_data.py -- pure-python logic (no torch / rdkit / pandas) for EXEMPLAR-CONDITIONED generation.

IDEA
  Instead of telling the generator "disease #512, target #977" (an ID with a learned vector that has no meaning for a
  disease or target the model has barely seen), we show it a few KNOWN LIGANDS of the context and ask for new molecules
  in that region of chemical space:
        target pool      T(t)     molecules listed under target t
        disease pool     D(d)     molecules listed under disease d
        pair pool        P(d, t)  molecules listed under BOTH d and t   (the most specific; it is how "the same target,
                                  different disease -> different chemistry" gets expressed)
  A new disease or target is "added to the library" simply by adding its known ligands to these pools; nothing has to be
  re-trained for it to be usable, and the better its pool, the better the conditioning ("as much as the data represents it").

THIS FILE
  label_maps      molecule -> diseases / targets / (disease, target) pairs (rows with too many labels are ignored)
  Library         the pools (built from the molecules that are ALLOWED -- i.e. not held out)
  EpisodeSampler  one training episode = (molecule x, exemplar molecules, exemplar roles); never contains x itself
  make_holdout    held-out WHOLE TARGETS (the "unseen target" test) + held-out molecules (the "unseen molecule" test)
  row_eval_items / heldout_target_items   fixed evaluation episodes with a TRUE and a SWAPPED exemplar set
  build_generation_exemplars   how a (disease, target) request turns into a SET of exemplars (a multi-exemplar rule; no script calls it now: the seed-and-edit generator
                               chooses its seeds in generate_for_request.py)
"""
import bisect
import random
from collections import defaultdict

# roles tell the model where an exemplar came from: no role, the target's pool, the disease's pool, or the (disease, target) pair's pool
ROLE_NONE, ROLE_TARGET, ROLE_DISEASE, ROLE_PAIR = 0, 1, 2, 3
# episode types, in the order of the probabilities given to EpisodeSampler
TYPES = ('pair', 'target', 'disease', 'mixed')


# Normalises a list of labels: strips them, drops non-strings and duplicates, sorts.  Returns a tuple, or () if there are none or more than max_list (a row that lists many
# labels says little about any one of them).
def _clean(values, max_list):
    vals = sorted({str(v).strip() for v in values if isinstance(v, str) and str(v).strip()})
    return tuple(vals) if 0 < len(vals) <= max_list else ()


# Builds the label tables from rows of the form dict(key, dis=[str], tgt=[str]).  A row contributes its diseases only if it lists between 1 and max_list of them; likewise
# for targets.  Pair labels come from rows that have usable diseases AND usable targets.  Returns (D, T, C): key -> set of diseases, key -> set of targets, key -> set of
# (disease, target) pairs.
def label_maps(rows, max_list=3):
    D, T, C = defaultdict(set), defaultdict(set), defaultdict(set)
    for r in rows:
        k = r['key']
        ds, ts = _clean(r.get('dis', ()), max_list), _clean(r.get('tgt', ()), max_list)
        D[k].update(ds)
        T[k].update(ts)
        for d in ds:
            for t in ts:
                C[k].add((d, t))
    return ({k: v for k, v in D.items() if v}, {k: v for k, v in T.items() if v}, {k: v for k, v in C.items() if v})


# Ligand pools built from the allowed (not held-out) molecules only.  pool_all[kind][label] is every such molecule with that label (used to pick the molecule to be generated);
# pool_ok[kind][label] is the subset that may be shown as an exemplar (ok(key) is true, for example because it is short enough).  kind is 't' (label = target), 'd' (label =
# disease) or 'dt' (label = (disease, target)).
class Library:

    # Drops the excluded molecules, builds the label tables and the sorted pools, and records which diseases occur at which target (and the reverse).
    def __init__(self, rows, max_list=3, exclude=frozenset(), ok=None):
        self.max_list = max_list
        self.exclude = frozenset(exclude)
        ok = ok or (lambda k: True)
        kept = [r for r in rows if r['key'] not in self.exclude]
        self.D, self.T, self.C = label_maps(kept, max_list)
        self.pool_all = {'t': defaultdict(set), 'd': defaultdict(set), 'dt': defaultdict(set)}
        for k, ts in self.T.items():
            for t in ts:
                self.pool_all['t'][t].add(k)
        for k, ds in self.D.items():
            for d in ds:
                self.pool_all['d'][d].add(k)
        for k, cs in self.C.items():
            for c in cs:
                self.pool_all['dt'][c].add(k)
        self.pool_all = {kind: {lab: sorted(v) for lab, v in m.items()} for kind, m in self.pool_all.items()}
        self.pool_ok = {kind: {lab: [k for k in v if ok(k)] for lab, v in m.items()} for kind, m in self.pool_all.items()}
        self.keys = sorted(set(self.D) | set(self.T))
        # diseases attested at a target, and targets at a disease, counted only where the pair pool has at least 3 usable exemplars
        self.dis_of_t, self.tgt_of_d = defaultdict(set), defaultdict(set)
        for (d, t), v in self.pool_ok['dt'].items():
            if len(v) >= 3:
                self.dis_of_t[t].add(d)
                self.tgt_of_d[d].add(t)

    # The molecules that may be used as exemplars for a label ([] if the label is unknown).
    def ok_pool(self, kind, label):
        return self.pool_ok[kind].get(label, [])

    # Every allowed molecule with a label ([] if the label is unknown).
    def all_pool(self, kind, label):
        return self.pool_all[kind].get(label, [])

    # Counts of molecules, targets, diseases, pairs, and pairs with at least 3 usable exemplars, for printing.
    def summary(self):
        return {'molecules': len(self.keys), 'targets': len(self.pool_all['t']), 'diseases': len(self.pool_all['d']),
                'pairs': len(self.pool_all['dt']),
                'pairs_ge3': sum(1 for v in self.pool_ok['dt'].values() if len(v) >= 3)}


# Draws up to k distinct molecules at random from cand, never x itself.  reject(x, c) returning True drops a candidate (for example a near-duplicate of x).  Only a random
# subset of 4k + 8 candidates is looked at, so large pools stay cheap; fewer than k may come back.
def draw(rng, cand, k, x=None, reject=None):
    cand = [c for c in cand if c != x]
    lim = 4 * k + 8
    pool = list(cand) if len(cand) <= lim else rng.sample(cand, lim)
    rng.shuffle(pool)
    out = []
    for c in pool:
        if reject is not None and x is not None and reject(x, c):
            continue
        out.append(c)
        if len(out) == k:
            break
    return out


# For molecule x within cand, returns (near, rest): near is up to `top` (similarity, molecule) pairs with similarity in [lo, hi], most similar first; rest is up to 300 other
# candidates with similarity <= hi.  A pool larger than cap is subsampled first.  The result is cached under `key`, so a subsample of a very large pool is fixed the first
# time it is used.
def _near_sets(rng, cand, x, sims, lo, hi, cap, top, cache, key):
    if cache is not None and key in cache:
        return cache[key]
    c2 = [c for c in cand if c != x]
    if len(c2) > cap:
        c2 = rng.sample(c2, cap)
    sv = sims(x, c2) if c2 else []
    near = sorted(((v, c) for v, c in zip(sv, c2) if lo <= v <= hi), reverse=True)[:top]
    nk = {c for _, c in near}
    rest = [c for v, c in zip(sv, c2) if v <= hi and c not in nk][:300]
    res = (near, rest)
    if cache is not None:
        cache[key] = res
    return res


# Draws k distinct molecules from cand (never x), of which n_near are NEAR x (similarity in [lo, hi]) and the rest random (similarity <= hi).  sims(x, cands) returns a
# list of similarities.  Returns None when x has no near neighbour in cand.  The near ones are chosen at random among the `top` most similar, so the model is not always
# shown the single best neighbour.
def draw_near(rng, cand, k, x, sims, n_near, lo, hi, cap=2000, top=5, cache=None, key=None):
    near, rest = _near_sets(rng, cand, x, sims, lo, hi, cap, max(top, n_near), cache, key)
    if not near:
        return None
    take = [c for _, c in rng.sample(near, min(n_near, len(near)))][:k]
    tk = set(take)
    others = [c for c in rest if c not in tk]
    rng.shuffle(others)
    out = take + others[:max(0, k - len(take))]
    rng.shuffle(out)
    return out


# Draws training episodes.  Groups (a target, a disease or a pair) are chosen with probability proportional to size ** balance_power, so rare contexts are seen far more
# often than their share of rows.  Episode types: 'pair' takes exemplars from the pair's pool, 'target' from the target's, 'disease' from the disease's, and 'mixed' some from
# the target's pool and the rest from the disease's.  The molecule to generate belongs to every pool its exemplars come from and is NEVER among the exemplars.
class EpisodeSampler:

    # Builds the group tables and the cumulative type probabilities.  near, if given, is {'sims': function(x, candidates) -> similarities, 'p': share of exemplar draws that
    # include a near neighbour of x, 'lo' and 'hi': the similarity window of a near neighbour (hi excludes near-duplicates), 'require': True to skip an episode whose molecule
    # has no near neighbour instead of falling back to random exemplars}.
    def __init__(self, lib, rng, k_max=6, type_probs=(0.35, 0.25, 0.15, 0.25), balance_power=0.5,
                 k_choices=(1, 2, 3, 3, 4, 4, 5, 5, 6, 6), reject=None, near=None):
        self.lib, self.rng, self.k_max, self.reject, self.near = lib, rng, k_max, reject, near
        self._near_cache = {}
        self.k_choices = [k for k in k_choices if k <= k_max] or [k_max]
        # cumulative probabilities, for drawing an episode type
        s = float(sum(type_probs))
        self.type_cum = []
        acc = 0.0
        for p in type_probs:
            acc += p / s
            self.type_cum.append(acc)
        # groups per type: (label, molecules) with at least 2 molecules, plus cumulative weights; a 'mixed' group is a pair whose target pool and disease pool both have at
        # least 2 usable exemplars
        self.groups = {}
        for typ, kind in (('pair', 'dt'), ('target', 't'), ('disease', 'd')):
            g = [(lab, keys) for lab, keys in sorted(lib.pool_all[kind].items(), key=lambda kv: str(kv[0])) if len(keys) >= 2]
            self.groups[typ] = self._weights(g, balance_power)
        g = [(lab, keys) for lab, keys in sorted(lib.pool_all['dt'].items(), key=lambda kv: str(kv[0]))
             if len(keys) >= 2 and len(lib.ok_pool('t', lab[1])) >= 2 and len(lib.ok_pool('d', lab[0])) >= 2]
        self.groups['mixed'] = self._weights(g, balance_power)

    # Turns a list of groups into (groups, cumulative weights proportional to size ** bp).
    @staticmethod
    def _weights(g, bp):
        cum, acc = [], 0.0
        for _, keys in g:
            acc += len(keys) ** bp
            cum.append(acc)
        return (g, cum)

    # Draws one group of the given type with probability proportional to its weight; None if the type has no groups.
    def _pick_group(self, typ):
        g, cum = self.groups[typ]
        if not g:
            return None
        u = self.rng.random() * cum[-1]
        return g[min(bisect.bisect_left(cum, u), len(g) - 1)]

    # Random number of exemplars for one episode.
    def _k(self):
        return self.rng.choice(self.k_choices)

    # Chooses the exemplars for molecule x from a pool.  With probability near['p'] the set contains 1 or 2 near neighbours of x (when x has any); if it has none and
    # near['require'] is set the result is empty, otherwise random exemplars are drawn.  Returns (exemplars, used_near).  ckey, if given, caches the near-neighbour search.
    def _pick(self, pool, k, x, ckey=None):
        nr = self.near
        if nr and self.rng.random() < nr['p']:
            ex = draw_near(self.rng, pool, k, x, nr['sims'], self.rng.choice((1, 1, 2)), nr['lo'], nr['hi'],
                           cache=self._near_cache if ckey is not None else None, key=(ckey, x) if ckey is not None else None)
            if ex:
                return ex, True
            if nr.get('require'):
                return [], False
        return draw(self.rng, pool, k, x, self.reject), False

    # Returns one training episode (a dict with type, molecule, disease, target, exemplars, roles and near flag), trying up to 30 times before giving up with None.
    def sample(self):
        for _ in range(30):
            u = self.rng.random()
            typ = TYPES[min(bisect.bisect_left(self.type_cum, u), len(TYPES) - 1)]
            ep = self._episode(typ)
            if ep:
                return ep
        return None

    # Builds one episode of the given type: picks a group, a molecule x from it and the exemplars from the matching pool(s).  A 'mixed' episode takes some exemplars from the
    # target's pool and the rest from the disease's.  Returns None when no exemplars could be found.
    def _episode(self, typ):
        grp = self._pick_group(typ)
        if grp is None:
            return None
        lab, keys = grp
        x = self.rng.choice(keys)
        lib = self.lib
        if typ == 'mixed':
            d, t = lab
            total = max(2, self._k())
            kt = self.rng.randint(1, total - 1)
            ex_t, n1 = self._pick(lib.ok_pool('t', t), kt, x, ('t', t))
            taken = set(ex_t)
            ex_d, n2 = self._pick([c for c in lib.ok_pool('d', d) if c not in taken], total - kt, x, None)
            if not ex_t or not ex_d:
                return None
            return {'type': 'mixed', 'key': x, 'd': d, 't': t, 'ex': ex_t + ex_d, 'near': n1 or n2,
                    'roles': [ROLE_TARGET] * len(ex_t) + [ROLE_DISEASE] * len(ex_d)}
        kind = {'pair': 'dt', 'target': 't', 'disease': 'd'}[typ]
        ex, used_near = self._pick(lib.ok_pool(kind, lab), self._k(), x, (kind, lab))
        if not ex:
            return None
        role = {'pair': ROLE_PAIR, 'target': ROLE_TARGET, 'disease': ROLE_DISEASE}[typ]
        d = lab[0] if typ == 'pair' else (lab if typ == 'disease' else None)
        t = lab[1] if typ == 'pair' else (lab if typ == 'target' else None)
        return {'type': typ, 'key': x, 'd': d, 't': t, 'ex': ex, 'roles': [role] * len(ex), 'near': used_near}


# Chooses what is held out, removed from EVERYTHING the model trains on (as molecule and as exemplar).  Held targets are whole targets: n_multi with at least 2 well
# populated diseases and n_plain others (named ones in hold_targets are taken first), each big enough.  Their molecules are split at random into 'lib' (the ligands a user
# would supply for a NEW target) and 'test' (what the generator should resemble); a target needs min_side of each.  Targets with several diseases also get per-disease lib /
# test sets, a disease contrast at a target the model never saw.  row_keys is a random share (row_frac) of the other molecules: unseen molecules at SEEN targets and diseases.
# Returns dict(held_targets, target_keys, row_keys, sets{t: {lib, test, lib_by_d, test_by_d}}).
def make_holdout(rows, rng, hold_targets=(), n_multi=8, n_plain=8, row_frac=0.08, max_list=3,
                 min_target_mols=12, multi_min_per_d=8, max_plain_size=150, lib_frac=0.5, ok=None, min_side=4):
    ok = ok or (lambda k: True)
    D, T, C = label_maps(rows, max_list)
    keys_of_t, keys_of_dt = defaultdict(set), defaultdict(set)
    for k, ts in T.items():
        for t in ts:
            keys_of_t[t].add(k)
    for k, cs in C.items():
        for c in cs:
            keys_of_dt[c].add(k)
    dis_of_t = defaultdict(list)
    for (d, t), ks in keys_of_dt.items():
        if len(ks) >= multi_min_per_d:
            dis_of_t[t].append(d)
    big = {t for t, ks in keys_of_t.items() if len(ks) >= min_target_mols}
    multi = sorted(t for t in big if len(dis_of_t[t]) >= 2)
    plain = sorted(t for t in big if t not in set(multi) and len(keys_of_t[t]) <= max_plain_size)
    chosen = [t for t in hold_targets if t in big]
    n_m = sum(1 for t in chosen if t in set(multi))
    rest_m = [t for t in multi if t not in chosen]
    rng.shuffle(rest_m)
    chosen += rest_m[:max(0, n_multi - n_m)]
    n_p = sum(1 for t in chosen if t in set(plain))
    rest_p = [t for t in plain if t not in chosen]
    rng.shuffle(rest_p)
    chosen += rest_p[:max(0, n_plain - n_p)]

    # every molecule of a held target is assigned once to 'lib' or 'test' (a molecule shared by several held targets keeps its first assignment)
    assign = {}
    for t in chosen:
        for k in sorted(keys_of_t[t]):
            if k not in assign:
                assign[k] = 'lib' if rng.random() < lib_frac else 'test'
    sets, kept = {}, []
    # a target is kept only if both sides have enough usable molecules; a per-disease set needs at least 3 on each side
    for t in chosen:
        lib = sorted(k for k in keys_of_t[t] if assign[k] == 'lib' and ok(k))
        test = sorted(k for k in keys_of_t[t] if assign[k] == 'test')
        if len(lib) < min_side or len(test) < min_side:
            continue
        lib_by_d, test_by_d = {}, {}
        for d in sorted(dis_of_t.get(t, ())):
            ks = keys_of_dt[(d, t)]
            ld = sorted(k for k in ks if assign.get(k) == 'lib' and ok(k))
            td = sorted(k for k in ks if assign.get(k) == 'test')
            if len(ld) >= 3 and len(td) >= 3:
                lib_by_d[d], test_by_d[d] = ld, td
        sets[t] = {'lib': lib, 'test': test, 'lib_by_d': lib_by_d, 'test_by_d': test_by_d}
        kept.append(t)
    # everything of the kept targets is withheld from training, plus a random share of all other molecules (row_keys)
    target_keys = sorted({k for t in kept for k in keys_of_t[t]})
    tk = set(target_keys)
    others = sorted((set(D) | set(T)) - tk)
    rng.shuffle(others)
    row_keys = sorted(others[:int(round(row_frac * len(others)))])
    return {'held_targets': kept, 'target_keys': target_keys, 'row_keys': row_keys, 'sets': sets}


# Random sample of up to k items from pool (all of it if the pool is smaller).
def _grab(rng, pool, k):
    return rng.sample(pool, min(k, len(pool)))


# Fixed evaluation items for held-out MOLECULES at seen targets and diseases.  labels is label_maps over ALL rows; lib is the training library (it contains none of row_keys).
# Each item holds, where available, a TRUE and a SWAPPED exemplar set: pair / pair_swap are the pair's pool against the same target's pool for another disease (tests the
# disease); tgt / tgt_swap are the target's pool against another target's (tests the target).  Each value is (exemplar keys, roles); the roles are None.
def row_eval_items(lib, row_keys, labels, rng, k=5, cap=500):
    Dl, Tl, Cl = labels
    out = []
    all_t = sorted(t for t, v in lib.pool_ok['t'].items() if len(v) >= 3)
    for x in sorted(row_keys):
        if len(out) >= cap:
            break
        item = {'key': x, 'pair': None, 'pair_swap': None, 'tgt': None, 'tgt_swap': None}
        cs = [c for c in sorted(Cl.get(x, ())) if len(lib.ok_pool('dt', c)) >= 3]
        if cs:
            d, t = rng.choice(cs)
            item['pair'] = (_grab(rng, lib.ok_pool('dt', (d, t)), k), None)
            alts = sorted(d2 for d2 in lib.dis_of_t.get(t, ()) if d2 not in Dl.get(x, ()))
            if alts:
                d2 = rng.choice(alts)
                item['pair_swap'] = (_grab(rng, lib.ok_pool('dt', (d2, t)), k), None)
            item['pair_ctx'] = (d, t)
        ts = [t for t in sorted(Tl.get(x, ())) if len(lib.ok_pool('t', t)) >= 3]
        if ts:
            t = rng.choice(ts)
            item['tgt'] = (_grab(rng, lib.ok_pool('t', t), k), None)
            # 'hard' swap targets share a disease with x, so the swap tests the target rather than the disease; if there are none, any other target is used
            hard = sorted({t2 for d in Dl.get(x, ()) for t2 in lib.tgt_of_d.get(d, ())
                           if t2 not in Tl.get(x, ()) and len(lib.ok_pool('t', t2)) >= 3})
            pool_t2 = hard or [t2 for t2 in all_t if t2 not in Tl.get(x, ())]
            if pool_t2:
                t2 = rng.choice(pool_t2)
                item['tgt_swap'] = (_grab(rng, lib.ok_pool('t', t2), k), None)
                item['tgt_swap_hard'] = bool(hard)
        if item['pair'] or item['tgt']:
            out.append(item)
    return out


# Fixed evaluation items for held-out WHOLE targets.  For each test molecule x of a held target t: 'true' is k molecules of t's lib and 'swap' k molecules of ANOTHER held
# target's lib (target test); for targets with at least 2 diseases, the disease items use k molecules of lib_by_d[d] against lib_by_d[d2] (disease test at an unseen target).
# Returns (target_items, disease_items), each a list of dicts {key, t, true, swap, ...} with (keys, None) values.
def heldout_target_items(sets, rng, k=5, cap_per_target=40):
    ts = [t for t in sorted(sets) if len(sets[t]['lib']) >= 3]
    t_items, d_items = [], []
    for i, t in enumerate(ts):
        others = [u for u in ts if u != t]
        s = sets[t]
        for x in s['test'][:cap_per_target]:
            it = {'key': x, 't': t, 'true': (_grab(rng, s['lib'], k), None), 'swap': None}
            if others:
                u = others[rng.randrange(len(others))]
                it['swap'] = (_grab(rng, sets[u]['lib'], k), None)
                it['t2'] = u
            t_items.append(it)
        ds = sorted(s['lib_by_d'])
        if len(ds) >= 2:
            for d in ds:
                for x in s['test_by_d'][d][:cap_per_target]:
                    d2 = rng.choice([e for e in ds if e != d])
                    d_items.append({'key': x, 't': t, 'd': d, 'd2': d2,
                                    'true': (_grab(rng, s['lib_by_d'][d], k), None),
                                    'swap': (_grab(rng, s['lib_by_d'][d2], k), None)})
    return t_items, d_items


# Turns a (disease, target) request into a set of exemplars.  If the pair has at least min_pair usable ligands: k of them (mode 'pair').  Otherwise, if both the target and the
# disease have ligands: about half from each (mode 'mixed').  With only one of them: k from that pool (mode 'target' or 'disease'); with neither: none (mode 'none').  A NEW
# disease or target works as soon as its ligands are in the pools.  Returns (keys, roles, mode).
def build_generation_exemplars(lib, disease, target, k=5, rng=None, min_pair=3):
    rng = rng or random.Random(0)
    pair = lib.ok_pool('dt', (disease, target)) if (disease and target) else []
    if len(pair) >= min_pair:
        ex = _grab(rng, pair, k)
        return ex, [ROLE_PAIR] * len(ex), 'pair'
    tp = lib.ok_pool('t', target) if target else []
    dp = lib.ok_pool('d', disease) if disease else []
    if tp and dp:
        kt = max(1, k // 2)
        ex_t = _grab(rng, tp, kt)
        ex_d = _grab(rng, [c for c in dp if c not in set(ex_t)], max(1, k - len(ex_t)))
        if ex_d:
            return ex_t + ex_d, [ROLE_TARGET] * len(ex_t) + [ROLE_DISEASE] * len(ex_d), 'mixed'
        return ex_t, [ROLE_TARGET] * len(ex_t), 'target'
    if tp:
        ex = _grab(rng, tp, k)
        return ex, [ROLE_TARGET] * len(ex), 'target'
    if dp:
        ex = _grab(rng, dp, k)
        return ex, [ROLE_DISEASE] * len(ex), 'disease'
    return [], [], 'none'
