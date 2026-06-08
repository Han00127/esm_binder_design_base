"""metrics.py — Phase 0.1: ESMFold2 출력용 인터페이스 평가기 (IPSAE / ipTM / pDockQ2).

Roland Dunbrack의 ipsae.py(v4, MIT) 코어 수식을 그대로 포팅하되,
ESMFold2 출력 포맷(cif + _pae.npy + _confidence.json)을 **네이티브로** 읽는다.
원본은 Boltz용 .npz/별도 plddt 파일을 가정하므로 그대로 적용 불가 → 이 모듈로 우회.

reference:
  ipSAE  : https://www.biorxiv.org/content/10.1101/2025.02.10.637595v2
  pDockQ2: Zhu et al., Bioinformatics 2023

사용:
  # CLI (한 케이스 평가)
  python metrics.py adalimumab_cdr_out/adalimumab_designed_structure_msa

  # 코드에서 (상관 실험 등 반복 호출)
  from metrics import evaluate_complex
  res = evaluate_complex(cif_path, pae_path, confidence_path)
  res["pairs"][("H","T")]["ipsae_max"]   # H↔T 인터페이스 ipSAE
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

PROTEIN_RES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}


# ── ptm / d0 ─────────────────────────────────────────────────────────────────
def ptm_func(x, d0):
    return 1.0 / (1.0 + (x / d0) ** 2.0)


def calc_d0(L, min_value=1.0):
    L = float(max(26.0, L))
    return max(min_value, 1.24 * (L - 15.0) ** (1.0 / 3.0) - 1.8)


def calc_d0_array(L, min_value=1.0):
    L = np.maximum(26.0, np.asarray(L, dtype=float))
    return np.maximum(min_value, 1.24 * (L - 15.0) ** (1.0 / 3.0) - 1.8)


# ── cif 파서 (ESMFold2 / Boltz 류 mmCIF, 필드 순서 무관) ─────────────────────
def parse_cif(cif_path: str):
    """반환: chains(N,), ca_coords(N,3), cb_coords(N,3), resnums(N,), resnames(N,).

    토큰(=잔기) 단위. 표준 단백질 잔기만(ligand/HETATM 비표준 제외).
    distance 는 원본과 동일하게 CB(GLY=CA) 기준.
    """
    fields: dict[str, int] = {}
    nfield = 0
    ca_by_res: dict[tuple, dict] = {}
    cb_by_res: dict[tuple, dict] = {}
    order: list[tuple] = []

    with open(cif_path) as fh:
        for line in fh:
            if line.startswith("_atom_site."):
                fields[line.strip().split(".")[1]] = nfield
                nfield += 1
                continue
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            t = line.split()
            comp = t[fields["label_comp_id"]]
            if comp not in PROTEIN_RES:
                continue
            atom = t[fields["label_atom_id"]]
            chain = t[fields.get("auth_asym_id", fields["label_asym_id"])]
            resnum = int(t[fields["label_seq_id"]])
            xyz = np.array([float(t[fields["Cartn_x"]]),
                            float(t[fields["Cartn_y"]]),
                            float(t[fields["Cartn_z"]])])
            key = (chain, resnum)
            rec = {"chain": chain, "resnum": resnum, "res": comp, "coor": xyz}
            if atom == "CA":
                if key not in ca_by_res:
                    order.append(key)
                ca_by_res[key] = rec
            # CB (GLY 은 CA 를 CB 대용으로)
            if atom == "CB" or (comp == "GLY" and atom == "CA"):
                cb_by_res[key] = rec

    chains, ca, cb, resnums, resnames = [], [], [], [], []
    for key in order:
        ca_rec = ca_by_res[key]
        cb_rec = cb_by_res.get(key, ca_rec)  # CB 없으면 CA 대용
        chains.append(ca_rec["chain"])
        ca.append(ca_rec["coor"])
        cb.append(cb_rec["coor"])
        resnums.append(ca_rec["resnum"])
        resnames.append(ca_rec["res"])
    return (np.array(chains), np.array(ca), np.array(cb),
            np.array(resnums), np.array(resnames))


# ── 메인 평가 ────────────────────────────────────────────────────────────────
def evaluate_complex(cif_path: str, pae_path: str, confidence_path: str,
                     pae_cutoff: float = 10.0, dist_cutoff: float = 10.0) -> dict:
    chains, _ca, cb, resnums, _resn = parse_cif(cif_path)
    numres = len(chains)

    pae = np.load(pae_path).astype(float)
    assert pae.shape == (numres, numres), \
        f"PAE {pae.shape} != residues {numres} (cif/pae 토큰 수 불일치)"

    conf = json.load(open(confidence_path))
    plddt = np.array(conf.get("plddt_per_token", np.zeros(numres)), dtype=float)
    if plddt.max() <= 1.0:                       # 0–1 → 0–100
        plddt = plddt * 100.0
    pcm = np.array(conf.get("pair_chains_iptm", []), dtype=float)  # (C,C) or empty

    # 체인 순서(첫 등장 순)
    _, first = np.unique(chains, return_index=True)
    uniq = list(chains[np.sort(first)])
    cidx = {c: i for i, c in enumerate(uniq)}

    # CB 거리행렬
    dist = np.sqrt(((cb[:, None, :] - cb[None, :, :]) ** 2).sum(-1))

    pairs: dict[tuple, dict] = {}
    for c1 in uniq:
        for c2 in uniq:
            if c1 == c2:
                continue
            m1 = chains == c1
            m2 = chains == c2

            # ── ipSAE / ipTM (asym: c1 의 잔기를 align 기준) ──
            n0chn = int(m1.sum() + m2.sum())
            d0chn = calc_d0(n0chn)
            ptm_d0chn = ptm_func(pae, d0chn)

            valid = np.outer(m1, m2) & (pae < pae_cutoff)   # (N,N)
            # n0dom: pae<cutoff 인 인터페이스 잔기 수(c1,c2 합집합)
            int_res1 = np.unique(np.where(valid.any(axis=1) & m1)[0])
            int_res2 = np.unique(np.where(valid.any(axis=0) & m2)[0])
            n0dom = len(int_res1) + len(int_res2)
            d0dom = calc_d0(n0dom) if n0dom > 0 else 1.0
            ptm_d0dom = ptm_func(pae, d0dom)

            n0res_row = valid.sum(axis=1)                   # 잔기별 good-pair 수
            d0res_row = calc_d0_array(n0res_row)

            rows = np.where(m1)[0]
            ipsae_d0res = np.zeros(numres)
            ipsae_d0chn = np.zeros(numres)
            ipsae_d0dom = np.zeros(numres)
            iptm_d0chn = np.zeros(numres)
            for i in rows:
                vp = valid[i]
                iptm_d0chn[i] = ptm_d0chn[i, m2].mean() if m2.any() else 0.0
                if vp.any():
                    ipsae_d0chn[i] = ptm_d0chn[i, vp].mean()
                    ipsae_d0dom[i] = ptm_d0dom[i, vp].mean()
                    ipsae_d0res[i] = ptm_func(pae[i], d0res_row[i])[vp].mean()

            pairs[(c1, c2)] = {
                "ipsae_asym": float(ipsae_d0res.max()),
                "ipsae_d0chn_asym": float(ipsae_d0chn.max()),
                "ipsae_d0dom_asym": float(ipsae_d0dom.max()),
                "iptm_d0chn_asym": float(iptm_d0chn.max()),
                "n0chn": n0chn, "n0dom": n0dom,
                "nres1": len(int_res1), "nres2": len(int_res2),
                "_ipsae_d0res_max_byres": ipsae_d0res,  # 내부용(max 계산)
            }

            # ── pDockQ2 (c1→c2) ──
            sum_ptm, npairs, iface = 0.0, 0, set()
            for i in rows:
                near = m2 & (dist[i] <= 8.0)
                if near.any():
                    npairs += int(near.sum())
                    sum_ptm += ptm_func(pae[i, near], 10.0).sum()
                    iface.add(i)
                    iface.update(np.where(near)[0].tolist())
            if npairs > 0:
                mean_plddt = plddt[list(iface)].mean()
                x = mean_plddt * (sum_ptm / npairs)
                pdockq2 = 1.31 / (1 + math.exp(-0.075 * (x - 84.733))) + 0.005
            else:
                pdockq2 = 0.0
            pairs[(c1, c2)]["pdockq2"] = float(pdockq2)

            # ── 모델 ipTM (pair_chains_iptm 에서) ──
            if pcm.size and c1 in cidx and c2 in cidx:
                pairs[(c1, c2)]["iptm_model"] = float(pcm[cidx[c1], cidx[c2]])

    # ── 대칭화: max(c1→c2, c2→c1) ──
    for c1 in uniq:
        for c2 in uniq:
            if c1 >= c2:
                continue
            a, b = pairs[(c1, c2)], pairs[(c2, c1)]
            for key in ("ipsae", "ipsae_d0chn", "ipsae_d0dom", "iptm_d0chn"):
                mx = max(a[f"{key}_asym"], b[f"{key}_asym"])
                a[f"{key}_max"] = b[f"{key}_max"] = mx
            for key in ("pdockq2",):
                mx = max(a[key], b[key])
                a[f"{key}_max"] = b[f"{key}_max"] = mx

    # 내부용 byres 제거
    for p in pairs.values():
        p.pop("_ipsae_d0res_max_byres", None)

    return {
        "iptm": float(conf.get("iptm", -1)),
        "ptm": float(conf.get("ptm", -1)),
        "plddt_mean": float(conf.get("plddt_mean", plddt.mean() / 100.0)),
        "chains": uniq,
        "pairs": pairs,
        "params": {"pae_cutoff": pae_cutoff, "dist_cutoff": dist_cutoff},
    }


def _resolve_stem(stem: str):
    stem = stem.replace("_pae.npy", "").replace("_confidence.json", "").replace(".cif", "")
    return (f"{stem}.cif", f"{stem}_pae.npy", f"{stem}_confidence.json")


def main(argv):
    if len(argv) < 2:
        print("usage: python metrics.py <output_stem | dir/name>  [pae_cutoff] [dist_cutoff]")
        print("  ex : python metrics.py adalimumab_cdr_out/adalimumab_designed_structure_msa")
        return 1
    cif, pae, conf = _resolve_stem(argv[1])
    pae_cut = float(argv[2]) if len(argv) > 2 else 10.0
    dist_cut = float(argv[3]) if len(argv) > 3 else 10.0
    for p in (cif, pae, conf):
        if not Path(p).exists():
            print(f"[ERROR] missing: {p}")
            return 1

    r = evaluate_complex(cif, pae, conf, pae_cut, dist_cut)
    print(f"\n=== {Path(cif).name} ===")
    print(f"global : ipTM={r['iptm']:.4f}  pTM={r['ptm']:.4f}  "
          f"pLDDT={r['plddt_mean']:.4f}  chains={r['chains']}")
    print(f"params : pae_cutoff={pae_cut}  dist_cutoff={dist_cut}\n")
    seen = set()
    hdr = f"{'pair':7} {'ipSAE':>8} {'ipSAE_chn':>10} {'ipSAE_dom':>10} " \
          f"{'ipTM_d0chn':>11} {'ipTM_model':>11} {'pDockQ2':>8} {'nres1':>6} {'nres2':>6}"
    print(hdr)
    print("-" * len(hdr))
    for (c1, c2), p in r["pairs"].items():
        key = frozenset((c1, c2))
        if key in seen:
            continue
        seen.add(key)
        print(f"{c1}-{c2:5} {p['ipsae_max']:8.4f} {p['ipsae_d0chn_max']:10.4f} "
              f"{p['ipsae_d0dom_max']:10.4f} {p['iptm_d0chn_max']:11.4f} "
              f"{p.get('iptm_model', float('nan')):11.4f} {p['pdockq2_max']:8.4f} "
              f"{p['nres1']:6d} {p['nres2']:6d}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
