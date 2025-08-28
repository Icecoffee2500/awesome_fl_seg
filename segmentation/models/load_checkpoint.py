import torch
from collections import defaultdict

def normalize_key(k):
    for p in ('module.', 'model.', 'state_dict.'):
        if k.startswith(p):
            return k[len(p):]
    return k

# 2D->4D adapt helpers
def expand_center(src, tgt_shape, device=None):
    out, inp = src.shape
    out_t, in_t, kH, kW = tgt_shape
    assert out == out_t and inp == in_t
    res = torch.zeros(tgt_shape, dtype=src.dtype, device=device)
    res[:,:,kH//2,kW//2] = src
    return res

def expand_tile(src, tgt_shape, device=None, normalize=False):
    out, inp = src.shape
    out_t, in_t, kH, kW = tgt_shape
    assert out == out_t and inp == in_t
    res = src.unsqueeze(-1).unsqueeze(-1).repeat(1,1,kH,kW).to(device)
    if normalize:
        res = res / (kH * kW)
    return res

def expand_avg(src, tgt_shape, device=None):
    # spread src evenly: each spatial gets src/(kH*kW)
    return expand_tile(src, tgt_shape, device=device, normalize=True)

# common suffix translation rules (target_suffix -> candidate_source_suffix)
COMMON_SUFFIX_MAP = {
    '.norm.g': '.norm.weight',
    '.norm.b': '.norm.bias',
    '.fn.to_out.weight': '.attn.attn.out_proj.weight',
    '.fn.to_out.bias': '.attn.attn.out_proj.bias',
    '.fn.net.0.weight': '.ffn.layers.0.weight',  # heuristic mapping for some FFN names
    '.fn.net.0.bias': '.ffn.layers.0.bias',
    '.fn.net.3.weight': '.ffn.layers.4.weight',
    '.fn.net.3.bias': '.ffn.layers.4.bias',
    # add more heuristics if you discover more patterns
}

def try_name_translations(tgt_key, src_state):
    """
    Return a list of candidate source keys by applying suffix translations.
    """
    candidates = []
    for tgt_suf, src_suf in COMMON_SUFFIX_MAP.items():
        if tgt_key.endswith(tgt_suf):
            prefix = tgt_key[:-len(tgt_suf)]
            candidates.append(prefix + src_suf)
    return candidates

def advanced_load_with_name_translation_and_expansion(model, ckpt_path,
                                                     only_backbone=True,
                                                     try_tile=True,
                                                     try_avg=False,
                                                     try_center=True,
                                                     verbose=True):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if isinstance(ckpt, dict):
        if 'state_dict' in ckpt:
            src_state = ckpt['state_dict']
        elif 'model' in ckpt:
            src_state = ckpt['model']
        else:
            src_state = ckpt
    else:
        raise RuntimeError("unexpected checkpoint format")

    src_state = {normalize_key(k): v for k, v in src_state.items()}
    if only_backbone:
        src_state = {k: v for k, v in src_state.items() if ('backbone' in k) or ('mit' in k) or ('encoder' in k)}

    tgt_state = model.state_dict()
    tgt_keys = list(tgt_state.keys())
    src_keys = list(src_state.keys())

    # First do previous splitting of in_proj -> q/kv (reuse earlier logic) 
    # (omitted here for brevity; assume you've already created '__split__...' entries if needed)
    # For safety, we will also try source candidates discovered via suffix map.

    new_state = {}
    used_src = set()

    # quick exact and suffix fuzzy mapping (like earlier)
    src_by_suffix = defaultdict(list)
    for k in src_keys:
        toks = k.split('.')
        for L in range(1, min(6, len(toks)+1)):
            src_by_suffix['.'.join(toks[-L:])].append(k)

    for t in tgt_keys:
        # 1) exact
        if t in src_state and src_state[t].shape == tgt_state[t].shape:
            new_state[t] = src_state[t].to(tgt_state[t].dtype)
            used_src.add(t)
            continue

        # 2) translated name candidates
        trans_cands = try_name_translations(t, src_state)
        found = False
        for cand in trans_cands:
            if cand in src_state and src_state[cand].shape == tgt_state[t].shape:
                new_state[t] = src_state[cand].to(tgt_state[t].dtype)
                used_src.add(cand)
                found = True
                break
        if found:
            continue

        # 3) suffix fuzzy (try longest suffix match)
        toks = t.split('.')
        for L in range(min(6, len(toks)), 0, -1):
            suf = '.'.join(toks[-L:])
            for c in src_by_suffix.get(suf, []):
                if c in used_src:
                    continue
                s = src_state[c]
                if s.shape == tgt_state[t].shape:
                    new_state[t] = s.to(tgt_state[t].dtype)
                    used_src.add(c)
                    found = True
                    break
            if found:
                break
        if found:
            continue

        # 4) try split placeholders created earlier (names like '__split__'+t)
        split_k = '__split__' + t
        if split_k in src_state:
            s = src_state[split_k]
            # adapt dims if needed below
        else:
            s = None

        # 5) adaptation for 2D->4D mismatches if we have a matching 2D source
        if s is None:
            # try find a 2D candidate with matching channel dims but shape mismatch
            # heuristics: find any src with same (out,in) if src is 2D and tgt is 4D
            if tgt_state[t].ndim == 4:
                out_t, in_t, kH, kW = tgt_state[t].shape
                # search src keys with shape (out_t, in_t)
                for c in src_keys:
                    if c in used_src:
                        continue
                    s_candidate = src_state[c]
                    if s_candidate.ndim == 2 and s_candidate.shape[0] == out_t and s_candidate.shape[1] == in_t:
                        # try expansions in order: center -> tile -> avg depending on flags
                        if try_center:
                            new_state[t] = expand_center(s_candidate, tuple(tgt_state[t].shape)).to(tgt_state[t].dtype)
                            used_src.add(c)
                            found = True
                            if verbose:
                                print(f"[adapt-center] {c} -> {t}")
                            break
                        if try_tile:
                            new_state[t] = expand_tile(s_candidate, tuple(tgt_state[t].shape), normalize=False).to(tgt_state[t].dtype)
                            used_src.add(c)
                            found = True
                            if verbose:
                                print(f"[adapt-tile] {c} -> {t}")
                            break
                        if try_avg:
                            new_state[t] = expand_avg(s_candidate, tuple(tgt_state[t].shape)).to(tgt_state[t].dtype)
                            used_src.add(c)
                            found = True
                            if verbose:
                                print(f"[adapt-avg] {c} -> {t}")
                            break
                if found:
                    continue

        else:
            # we have a split source 's' (1D/2D); try to adapt if shapes differ
            tgt = tgt_state[t]
            if s.shape == tgt.shape:
                new_state[t] = s.to(tgt.dtype)
                continue
            if s.ndim == 2 and tgt.ndim == 4:
                # try chosen expansion methods
                if try_center:
                    new_state[t] = expand_center(s, tuple(tgt.shape)).to(tgt.dtype)
                    if verbose: print(f"[adapt-split-center] {split_k} -> {t}")
                    continue
                if try_tile:
                    new_state[t] = expand_tile(s, tuple(tgt.shape), normalize=False).to(tgt.dtype)
                    if verbose: print(f"[adapt-split-tile] {split_k} -> {t}")
                    continue
                if try_avg:
                    new_state[t] = expand_avg(s, tuple(tgt.shape)).to(tgt.dtype)
                    if verbose: print(f"[adapt-split-avg] {split_k} -> {t}")
                    continue

    # load non-strict
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if verbose:
        print("=== advanced load summary ===")
        print("Total target params:", len(tgt_keys))
        print("Total source params:", len(src_keys))
        print("Mapped/adapted params:", len(new_state))
        print("Torch missing keys:", len(missing))
        print("Torch unexpected keys:", len(unexpected))
    return new_state, missing, unexpected
