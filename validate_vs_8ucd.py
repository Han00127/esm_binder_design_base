"""validate_vs_8ucd.py — 예측 복합체를 8UCD에 정렬해 *구조기반 PAE*(frame-based aligned error)로
ipSAE 계산. 추가로 모델 자기보고 PAE(confidences.json)로도 ipSAE 계산해 비교.

  구조기반: 모델이 8UCD 실험구조를 얼마나 재현했나(정답 대비 오차).
  모델PAE : 모델이 스스로 보고한 신뢰도.
  → 둘 차이 = 모델 confidence calibration (구조는 맞췄는데 자신 없으면 모델PAE가 더 낮음).

정렬: ① 체인별 서열정렬(항체 Fv부분만, 설계는 CDR 서열 달라도 골격이 등록 고정) ② trimer 6순열을
항체포함 superpose해 RMSD 최소(=항체 일관) 채택 ③ frame-based PAE, 미매핑=30.

사용: PYTHONPATH=. python validate_vs_8ucd.py [pred1.cif pred2.cif ...]
  각 cif 옆의 {stem}_confidences.json 자동 사용. 인자 없으면 기본 top1.
"""
from __future__ import annotations

import glob
import itertools
import json
import os
import sys

import numpy as np
from Bio.Align import PairwiseAligner, substitution_matrices

import metrics

TRUE = "runs/8UCD.cif"
OUTDIR = "runs/_vs8ucd"
DEFAULT = "runs/20260619_steap1/ipsae_top1_seed-1557840021_sample-2.cif"
AG, AB = ("A", "B", "C"), ("H", "L")
UNMAP_PAE = 30.0
AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
       "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
       "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}

_ALN = PairwiseAligner()
_ALN.mode = "global"
_ALN.substitution_matrix = substitution_matrices.load("BLOSUM62")
_ALN.open_gap_score = -11
_ALN.extend_gap_score = -1


def parse_bb(path):
    fields, nf, res = {}, 0, {}
    for line in open(path):
        if line.startswith("_atom_site."):
            fields[line.strip().split(".")[1]] = nf; nf += 1; continue
        if not line.startswith("ATOM"):
            continue
        t = line.split()
        atom = t[fields["label_atom_id"]]
        if atom not in ("N", "CA", "C"):
            continue
        comp = t[fields["label_comp_id"]]
        if comp not in AA3:
            continue
        ch = t[fields.get("auth_asym_id", fields["label_asym_id"])]
        rn = int(t[fields["label_seq_id"]])
        xyz = np.array([float(t[fields["Cartn_x"]]), float(t[fields["Cartn_y"]]),
                        float(t[fields["Cartn_z"]])])
        res.setdefault((ch, rn), {"aa": AA3[comp]})[atom] = xyz
    return {k: v for k, v in res.items() if all(a in v for a in ("N", "CA", "C"))}


def chain_residues(bb, chain):
    return sorted([(rn, bb[(c, rn)]["aa"]) for (c, rn) in bb if c == chain])


def align_map(resA, resB):
    sA = "".join(a for _, a in resA); sB = "".join(a for _, a in resB)
    if not sA or not sB:
        return {}, 0, 0
    aln = _ALN.align(sA, sB)[0]
    idxA = [rn for rn, _ in resA]; idxB = [rn for rn, _ in resB]
    mp, ident, n = {}, 0, 0
    for (a0, a1), (b0, b1) in zip(aln.aligned[0], aln.aligned[1]):
        for k in range(a1 - a0):
            ia, ib = a0 + k, b0 + k
            mp[idxA[ia]] = idxB[ib]; n += 1
            if sA[ia] == sB[ib]:
                ident += 1
    return mp, (ident / n if n else 0), n


def kabsch_rmsd(P, Q):
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    V, S, Wt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(V @ Wt))
    R = V @ np.diag([1, 1, d]) @ Wt
    return float(np.sqrt(((Pc @ R - Qc) ** 2).sum(1).mean()))


def frame(rec):
    N, CA, C = rec["N"], rec["CA"], rec["C"]
    e1 = N - CA; e1 /= np.linalg.norm(e1) + 1e-8
    u = C - CA; e2 = u - (u @ e1) * e1; e2 /= np.linalg.norm(e2) + 1e-8
    e3 = np.cross(e1, e2)
    return np.stack([e1, e2, e3], axis=1)


def _binder(ev):
    return {ag: round(max((ev["pairs"].get((ab, ag)) or ev["pairs"].get((ag, ab)) or {})
                          .get("ipsae_max", 0) for ab in AB), 3) for ag in AG}


_BBT = None


def struct_based(pred_cif, out_pae, out_conf, verbose=False):
    """8UCD 정렬 → frame-based PAE → ipSAE. 반환 (binder dict, rmsd, coverage)."""
    global _BBT
    if _BBT is None:
        _BBT = parse_bb(TRUE)
    bbP, bbT = parse_bb(pred_cif), _BBT
    corr = {}
    for ab in AB:                                       # 항체 Fv (CDR 달라도 골격이 등록)
        mp, ident, n = align_map(chain_residues(bbP, ab), chain_residues(bbT, ab))
        for rnP, rnT in mp.items():
            corr[(ab, rnP)] = (ab, rnT)
        if verbose:
            print(f"    항체 {ab}: 매핑 {n}, 동일성 {ident:.0%}")
    abP = np.array([bbP[k]["CA"] for k in corr]); abT = np.array([bbT[v]["CA"] for v in corr.values()])
    seqmaps = {(x, y): align_map(chain_residues(bbP, x), chain_residues(bbT, y))[0]
               for x in AG for y in AG}
    best = None
    for perm in itertools.permutations(AG):
        P, Q = [abP], [abT]
        for x, y in zip(AG, perm):
            for rnP, rnT in seqmaps[(x, y)].items():
                P.append(bbP[(x, rnP)]["CA"][None]); Q.append(bbT[(y, rnT)]["CA"][None])
        rmsd = kabsch_rmsd(np.concatenate(P), np.concatenate(Q))
        if best is None or rmsd < best[0]:
            best = (rmsd, perm)
    rmsd, perm = best
    for x, y in zip(AG, perm):
        for rnP, rnT in seqmaps[(x, y)].items():
            corr[(x, rnP)] = (y, rnT)

    chains, _, _, resnums, _ = metrics.parse_cif(pred_cif)
    order = list(zip(chains.tolist(), resnums.tolist()))
    Nn = len(order)
    CAp = np.full((Nn, 3), np.nan); CAt = np.full((Nn, 3), np.nan)
    Rp = np.zeros((Nn, 3, 3)); Rt = np.zeros((Nn, 3, 3)); mapped = np.zeros(Nn, bool)
    for k, key in enumerate(order):
        if key not in bbP:
            continue
        CAp[k] = bbP[key]["CA"]; Rp[k] = frame(bbP[key])
        if key in corr and corr[key] in bbT:
            CAt[k] = bbT[corr[key]]["CA"]; Rt[k] = frame(bbT[corr[key]]); mapped[k] = True
    pae = np.full((Nn, Nn), UNMAP_PAE)
    mi = np.where(mapped)[0]
    for i in mi:
        lp = (CAp[mi] - CAp[i]) @ Rp[i]
        lt = (CAt[mi] - CAt[i]) @ Rt[i]
        pae[i, mi] = np.linalg.norm(lp - lt, axis=1)
    np.fill_diagonal(pae, 0.0)
    np.save(out_pae, pae.astype(np.float32))
    json.dump({"plddt_per_token": [0.0] * Nn}, open(out_conf, "w"))
    ev = metrics.evaluate_complex(pred_cif, out_pae, out_conf)
    return _binder(ev), round(rmsd, 2), round(float(mapped.mean()), 2)


def _blocks(seq):
    out = []
    for c in seq:
        if out and out[-1][0] == c:
            out[-1][1] += 1
        else:
            out.append([c, 1])
    return out


def model_based(pred_cif, conf_json, out_pae, out_conf):
    """confidences.json 의 pae → ipSAE. 토큰 순서가 cif 와 동일(블록크기 일치)함을 검증 후 직접 사용."""
    chains, _, _, resnums, _ = metrics.parse_cif(pred_cif)
    d = json.load(open(conf_json))
    pae = np.array(d["pae"], dtype=np.float32)
    tch = d["token_chain_ids"]
    cb = [n for _, n in _blocks(chains.tolist())]
    tb = [n for _, n in _blocks(tch)]
    assert pae.shape[0] == len(chains) and cb == tb, f"토큰 불일치 cif{cb} vs conf{tb}"
    np.save(out_pae, pae)
    json.dump({"plddt_per_token": [0.0] * len(chains)}, open(out_conf, "w"))
    ev = metrics.evaluate_complex(pred_cif, out_pae, out_conf)
    return _binder(ev)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    cifs = sys.argv[1:] or [DEFAULT]
    print(f"{'설계':42} {'RMSD':>5} {'cov':>4} | {'구조기반(vs8UCD)':>22} | {'모델PAE':>20}")
    print("-" * 105)
    for cif in cifs:
        stem = os.path.splitext(os.path.basename(cif))[0]
        conf = os.path.join(os.path.dirname(cif), stem + "_confidences.json")
        sb, rmsd, cov = struct_based(cif, f"{OUTDIR}/{stem}_spae.npy", f"{OUTDIR}/{stem}_sconf.json")
        try:
            mb = model_based(cif, conf, f"{OUTDIR}/{stem}_mpae.npy", f"{OUTDIR}/{stem}_mconf.json") \
                if os.path.exists(conf) else None
        except Exception as e:
            mb = None; print("  model_based err:", str(e)[:60])
        sbs = f"A:{sb['A']} B:{sb['B']} C:{sb['C']}→{max(sb.values()):.3f}"
        mbs = (f"A:{mb['A']} B:{mb['B']} C:{mb['C']}→{max(mb.values()):.3f}") if mb else "없음"
        print(f"{stem[:42]:42} {rmsd:5.2f} {cov:4.2f} | {sbs:>22} | {mbs:>20}")


if __name__ == "__main__":
    main()
