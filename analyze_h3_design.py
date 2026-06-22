"""analyze_h3_design.py — H3-fix 설계가 나오는 대로 자동 분석.
  ① native 타겟 H3 대비 화학성 차이  ② 자연분포(H3__14) 대비  ③ loss 궤적 진단.
설계 로그(runs/h3fix_design.log)를 파싱해 완료된 trajectory마다 리포트를 runs/h3fix_analysis.txt 에 갱신.

사용: python analyze_h3_design.py [design.log] [out.txt]
"""
from __future__ import annotations

import re
import sys

import numpy as np

NATIVE = "TRWGYYGTRGYFNV"          # native(타겟) H3 (len14)
FIX = 6                            # 앞 6 고정, 뒤 8 설계
AROM = set("WYF")
ORDER = "ARNDCQEGHILKMFPSTWYV"
PROP = {'A': '소수성', 'R': '+전하', 'N': '극성', 'D': '-전하', 'C': 'SS', 'Q': '극성', 'E': '-전하',
        'G': '유연', 'H': '약+/방향족', 'I': '소수성', 'L': '소수성', 'K': '+전하', 'M': '소수성',
        'F': '방향족', 'P': '강직', 'S': '극성', 'T': '극성', 'W': '방향족', 'Y': '방향족', 'V': '소수성'}
H3PSSM = np.asarray(np.load("data/length_pssm_full.npz", allow_pickle=True)["H3__14"], float)  # [14,20]


def parse_log(path):
    trajs, cur = [], None
    for line in open(path):
        m = re.search(r"trajectory (\d+)/\d+ \(seed=(-?\d+)\)", line)
        if m:
            cur = {"traj": int(m.group(1)), "seed": int(m.group(2)), "ks": [], "final": None}
            trajs.append(cur); continue
        if cur is None:
            continue
        k = re.search(r"\[k=\s*(\d+)\].*inter=([\d.]+).*glob=([\d.]+).*comp=([\d.]+).*disto_iptm=([\d.]+)", line)
        if k:
            cur["ks"].append({"k": int(k.group(1)), "inter": float(k.group(2)), "glob": float(k.group(3)),
                              "comp": float(k.group(4)), "disto": float(k.group(5))}); continue
        f = re.search(r"disto_iptm=([\d.]+) arom=([\d.]+) nat_nll=([\d.]+) \(native=([\d.]+)\) CDR=(\w+)", line)
        if f:
            cur["final"] = {"disto": float(f.group(1)), "arom": float(f.group(2)),
                            "nat_nll": float(f.group(3)), "native_nll": float(f.group(4)), "cdr": f.group(5)}
    return trajs


def analyze(t):
    out, des = [], t["final"]["cdr"]
    out.append(f"\n{'='*78}\n■ trajectory {t['traj']} (seed={t['seed']})  CDR={des}")
    out.append(f"  disto_iptm={t['final']['disto']:.3f}  arom={t['final']['arom']:.2f}  "
               f"nat_nll={t['final']['nat_nll']:.2f} (native={t['final']['native_nll']:.2f})  "
               f"{'⚠부자연(>native)' if t['final']['nat_nll']>t['final']['native_nll'] else '✓자연수준'}")

    # ① 화학성 (설계영역만)
    out.append("  ① 화학성 (설계 102-109):")
    chg = []
    for i in range(FIX, 14):
        n, d = NATIVE[i], des[i]
        if n == d:
            continue
        note = []
        if n in AROM and d not in AROM: note.append("방향족상실")
        if n not in AROM and d in AROM: note.append("방향족추가")
        if n in "RK" and d not in "RK": note.append("+전하상실")
        if d == 'P': note.append("Pro강직")
        chg.append(f"{96+i}:{n}→{d}({PROP[n]}→{PROP[d]}{' '+'/'.join(note) if note else ''})")
    out.append("     " + (";  ".join(chg) if chg else "(native와 동일)"))
    na = sum(c in AROM for c in NATIVE[FIX:]); da = sum(c in AROM for c in des[FIX:])
    nr = sum(c in "RK" for c in NATIVE[FIX:]); dr = sum(c in "RK" for c in des[FIX:])
    out.append(f"     방향족 {na}→{da}개, +전하 {nr}→{dr}개")

    # ② 자연분포 대비
    out.append("  ② 자연분포(H3__14) 대비  [pos 설계(확률) vs 자연top]:")
    rare = []
    for i in range(FIX, 14):
        row = H3PSSM[i]; d = des[i]
        dp = row[ORDER.index(d)] if d in ORDER else 0
        top = sorted(zip(ORDER, row), key=lambda x: -x[1])[:2]
        tops = " ".join(f"{a}:{p:.2f}" for a, p in top)
        flag = " ←드묾" if dp < 0.05 else ""
        out.append(f"     {96+i} {d}({dp:.2f}) vs [{tops}]{flag}")
        if dp < 0.05:
            rare.append(f"{96+i}{d}(자연top {top[0][0]}:{top[0][1]:.2f})")
    if rare:
        out.append(f"     ⚠ 부자연 위치: {', '.join(rare)}")

    # ③ loss 궤적
    ks = t["ks"]
    if ks:
        inter = [x["inter"] for x in ks]; disto = [x["disto"] for x in ks]
        comp = [x["comp"] for x in ks]; glob = [x["glob"] for x in ks]
        imin_i = int(np.argmin(inter))
        out.append("  ③ loss 궤적:")
        out.append(f"     inter {inter[0]:.2f}→최저 {min(inter):.2f}(k={ks[imin_i]['k']})→끝 {inter[-1]:.2f}"
                   f"  {'⚠반등(미수렴)' if inter[-1] > min(inter)+0.02 else '수렴'}")
        out.append(f"     disto_iptm {disto[0]:.2f}→최고 {max(disto):.2f}→끝 {disto[-1]:.2f}"
                   f"  {'(floating ~0.2)' if max(disto) < 0.35 else ''}")
        out.append(f"     comp {comp[0]:.2f}→{comp[-1]:.2f}  glob {glob[0]:.3f}(변화 {max(glob)-min(glob):.3f})"
                   f"  {'⚠glob 무력(H3만 mutable)' if max(glob)-min(glob) < 0.01 else ''}")
    return "\n".join(out)


def main():
    log = sys.argv[1] if len(sys.argv) > 1 else "runs/h3fix_design.log"
    outp = sys.argv[2] if len(sys.argv) > 2 else "runs/h3fix_analysis.txt"
    trajs = [t for t in parse_log(log) if t["final"]]
    lines = [f"H3-fix 설계 분석 — 완료 {len(trajs)}개  (native 타겟 H3={NATIVE}, 고정 TRWGYY)"]
    # 요약 테이블
    lines.append(f"\n{'traj':>4} {'CDR':16} {'disto':>6} {'nat_nll':>7} {'arom':>5}  부자연위치수")
    for t in trajs:
        nrare = sum(1 for i in range(FIX, 14)
                    if (H3PSSM[i][ORDER.index(t['final']['cdr'][i])] if t['final']['cdr'][i] in ORDER else 0) < 0.05)
        lines.append(f"{t['traj']:>4} {t['final']['cdr']:16} {t['final']['disto']:>6.3f} "
                     f"{t['final']['nat_nll']:>7.2f} {t['final']['arom']:>5.2f}  {nrare}")
    for t in trajs:
        lines.append(analyze(t))
    open(outp, "w").write("\n".join(lines))
    print(f"[analyze] {len(trajs)}개 설계 분석 → {outp}")


if __name__ == "__main__":
    main()
