"""build_lmap_targets.py — 8UCD 실험구조에서 map-anchored loss(L_map)의 입력 추출.

설계 distogram 좌표계(antigen 759 concat + scFv 242, sc_off=759)로:
  ① interface 접촉쌍: 8UCD에서 CDR잔기↔항원잔기 (CB-CB<cutoff) → 실제 CA-CA 거리(=distogram 타깃).
     CSV의 side-chain 거리(2.6Å)가 아니라 *구조에서 잰 CA-CA*를 씀.
  ② H3 loop 내부 형태: H3 잔기들의 pairwise CA-CA (non-interface form 유지 타깃).

인덱싱:
  antigen global = {A:0, B:253, C:506}[chain] + crop_local   (crop_local = 8UCD resnum 정렬 위치)
  scFv  global = 759 + scFv_local   (VH: 0..120, VL: 136..241)

출력: data/lmap_targets.npz  +  사람이 읽는 표.
사용: PYTHONPATH=. python build_lmap_targets.py
"""
from __future__ import annotations

import numpy as np
import yaml

import metrics
from validate_vs_8ucd import align_map

TRUE = "runs/8UCD.cif"
CFG = "configs/full_steap1_trimer_fv.yaml"
SC_OFF = 759
CHAIN_OFF = {"A": 0, "B": 253, "C": 506}
VOFF = 121 + 15                      # scFv 내 VL 시작 (VH121 + linker15)
H3_LOCAL = list(range(96, 110))      # H3 scFv-local 위치 (고정 96-101 / 설계 102-109)
FIX_N = 6                            # H3 앞 6 고정
CONTACT_CB = 9.5                     # 접촉 판정 CB-CB 컷오프(Å)
AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
       "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
       "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}


def load_8ucd():
    ch, ca, cb, rn, rnm = metrics.parse_cif(TRUE)
    rec = {}                                              # (chain,resnum) -> dict
    perchain = {}                                         # chain -> [(resnum,aa)]
    for i in range(len(ch)):
        c, r = ch[i], int(rn[i])
        rec[(c, r)] = {"ca": ca[i], "cb": cb[i], "aa": AA3.get(rnm[i], "X")}
        perchain.setdefault(c, []).append((r, AA3.get(rnm[i], "X")))
    for c in perchain:
        perchain[c] = sorted(perchain[c])
    return rec, perchain


def main():
    cfg = yaml.safe_load(open(CFG))
    _cl = lambda s: s.replace(" ", "").replace("\n", "")
    vh = _cl(cfg["antibody"]["heavy"]["vh_sequence"])
    vl = _cl(cfg["antibody"]["light"]["vl_sequence"])
    crop = _cl(cfg["chains"][0]["sequence"])             # A/B/C 동일 crop

    rec, pc = load_8ucd()

    # ── 8UCD 항체 resnum → scFv_local 매핑 (H→VH, L→VL) ──
    vh_res = [(i, a) for i, a in enumerate(vh)]          # crop처럼 위치-기반 (resnum=위치)
    vl_res = [(i, a) for i, a in enumerate(vl)]
    mH, idH, _ = align_map(pc["H"], vh_res)              # 8UCD H resnum -> VH 위치
    mL, idL, _ = align_map(pc["L"], vl_res)              # 8UCD L resnum -> VL 위치
    ab_local = {}                                        # (8UCD체인,resnum) -> scFv_local
    for r8, vpos in mH.items():
        ab_local[("H", r8)] = vpos                       # VH: scFv_local = VH 위치
    for r8, vpos in mL.items():
        ab_local[("L", r8)] = VOFF + vpos                # VL: scFv_local = 136 + VL 위치
    print(f"[항체정렬] H {len(mH)}잔기({idH:.0%}), L {len(mL)}잔기({idL:.0%})")

    # ── 8UCD 항원 resnum → crop_local (각 protomer) ──
    crop_res = [(i, a) for i, a in enumerate(crop)]
    ag_local = {}                                        # (체인,resnum) -> crop_local
    for c in ("A", "B", "C"):
        m, idc, _ = align_map(pc[c], crop_res)
        for r8, cl in m.items():
            ag_local[(c, r8)] = cl
        print(f"[항원정렬] {c} {len(m)}잔기({idc:.0%})")

    # ── interface 접촉(CB-CB<cutoff) → CA-CA 타깃 ──
    rows = []
    for (abc, abr), sl in ab_local.items():
        abrec = rec[(abc, abr)]
        for (agc, agr), cl in ag_local.items():
            agrec = rec[(agc, agr)]
            dcb = float(np.linalg.norm(abrec["cb"] - agrec["cb"]))
            if dcb > CONTACT_CB:
                continue
            dca = float(np.linalg.norm(abrec["ca"] - agrec["ca"]))
            b_idx = SC_OFF + sl
            a_idx = CHAIN_OFF[agc] + cl
            region = ("H3-설계" if sl in H3_LOCAL[FIX_N:] else
                      "H3-고정" if sl in H3_LOCAL[:FIX_N] else
                      "VH" if sl < VOFF else "VL")
            rows.append({"b_idx": b_idx, "a_idx": a_idx, "ca_ca": round(dca, 1), "cb_cb": round(dcb, 1),
                         "ab": f"{abc}{abr}{abrec['aa']}", "sl": sl, "region": region,
                         "ag": f"{agc}:{agrec['aa']}{agr}", "ag_resnum": agr})

    rows.sort(key=lambda r: (r["region"] != "H3-설계", r["region"] != "H3-고정", r["sl"], r["ca_ca"]))

    # ── 출력 ──
    print(f"\n=== interface 접촉 {len(rows)}쌍 (CB-CB<{CONTACT_CB}Å) | CA-CA = distogram 타깃 ===")
    print(f"{'binder(ab)':14}{'sl':>4} {'region':8} {'→ antigen':10} {'CA-CA':>6} {'CB-CB':>6}  (b_idx,a_idx)")
    for r in rows:
        print(f"{r['ab']:14}{r['sl']:>4} {r['region']:8} {r['ag']:10} {r['ca_ca']:>6} {r['cb_cb']:>6}  "
              f"({r['b_idx']},{r['a_idx']})")

    # H3-설계 영역 접촉 요약(핵심)
    h3d = [r for r in rows if r["region"] == "H3-설계"]
    h3f = [r for r in rows if r["region"] == "H3-고정"]
    print(f"\n핵심: H3-고정 접촉 {len(h3f)}쌍, H3-설계 접촉 {len(h3d)}쌍")

    # ── H3 loop 내부 CA-CA (form 타깃) ──
    h3_glob_to_8ucd = {}                                 # scFv_local -> 8UCD (체인,resnum)
    for (abc, abr), sl in ab_local.items():
        if sl in H3_LOCAL:
            h3_glob_to_8ucd[sl] = (abc, abr)
    h3_sl = sorted(h3_glob_to_8ucd)
    h3_ca = np.array([rec[h3_glob_to_8ucd[sl]]["ca"] for sl in h3_sl])
    h3_intra = np.linalg.norm(h3_ca[:, None, :] - h3_ca[None, :, :], axis=-1)
    print(f"\n=== H3 loop 내부 CA-CA (form 타깃) — {len(h3_sl)}잔기 ===")
    print("  scFv_local:", h3_sl)
    print("  양 끝(96↔109) CA-CA:", round(float(h3_intra[0, -1]), 1), "Å (loop 폭)")

    # ── 저장 ──
    np.savez("data/lmap_targets.npz",
             inter_b_idx=np.array([r["b_idx"] for r in rows]),
             inter_a_idx=np.array([r["a_idx"] for r in rows]),
             inter_target_caca=np.array([r["ca_ca"] for r in rows], dtype=float),
             inter_region=np.array([r["region"] for r in rows]),
             h3_scfv_local=np.array(h3_sl),
             h3_intra_caca=h3_intra.astype(float))
    print("\n저장: data/lmap_targets.npz (inter 접촉쌍 + H3 loop 내부형태)")


if __name__ == "__main__":
    main()
