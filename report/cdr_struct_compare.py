"""CDR 구조 비교: native(1N8Z) vs s108(ESMFold2).
- 도메인(VH/VL) framework Cα로 superpose → CDR별 Cα RMSD, backbone(N,CA,C,O) RMSD,
  side-chain heavy-atom RMSD(동일 AA 위치만).
- CDR 잔기별 phi/psi 테이블 + Δ.
- 3D 오버레이(matplotlib) + Ramachandran 산점도 + PyMOL 스크립트.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import biotite.structure.io.pdbx as pdbx
import biotite.structure as struc
from biotite.sequence import ProteinSequence
from biotite.sequence.align import SubstitutionMatrix, align_optimal

os.chdir("/home/kyeongtak/structure_projects/esm_binder_design_base")

VH = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
VL = "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK"
CDR = {"H1": (27, 35, "VH"), "H2": (50, 62, "VH"), "H3": (98, 109, "VH"),
       "L1": (23, 34, "VL"), "L2": (49, 56, "VL"), "L3": (88, 97, "VL")}
DOMSEQ = {"VH": VH, "VL": VL}
NATIVE_CIF = "runs/rank_ce_out/1N8Z.cif"
S108_CIF = "runs/rank_ce_out/s108_exp_full_2021.cif"
MATRIX = SubstitutionMatrix.std_protein_matrix()


def load(path):
    f = pdbx.CIFFile.read(path)
    arr = pdbx.get_structure(f, model=1)
    return arr[struc.filter_amino_acids(arr)]


def chain_seq(chain):
    rids, rnames = struc.get_residues(chain)
    seq = ""
    for rn in rnames:
        try:
            seq += ProteinSequence.convert_letter_3to1(rn)
        except Exception:
            seq += "X"
    return seq, rids


def map_canon_to_resid(canon, chain):
    """canon 서열 각 위치 → chain res_id (local alignment, framework로 앵커)."""
    cseq, rids = chain_seq(chain)
    cseq_clean = cseq.replace("X", "A")
    ali = align_optimal(ProteinSequence(canon), ProteinSequence(cseq_clean),
                        MATRIX, gap_penalty=(-10, -1), local=True)[0]
    m = {}
    for a, b in ali.trace:
        if a >= 0 and b >= 0:
            m[a] = int(rids[b])
    return m


def atom(chain, rid, name):
    sel = chain[(chain.res_id == rid) & (chain.atom_name == name)]
    return sel.coord[0] if sel.array_length() > 0 else None


def sidechain_atoms(chain, rid):
    bb = np.isin(chain.atom_name, ["N", "CA", "C", "O"])
    sel = chain[(chain.res_id == rid) & (~bb) & (chain.element != "H")]
    return {a: c for a, c in zip(sel.atom_name, sel.coord)}


def kabsch(P, Q):
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = Q.mean(0) - R @ P.mean(0)
    return R, t


def applyT(R, t, X):
    return (R @ np.atleast_2d(X).T).T + t


def dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.degrees(np.arctan2(y, x))


def phipsi(chain, rid_sorted, idx):
    """rid_sorted: 도메인 잔기 res_id 정렬. idx 위치 잔기의 phi/psi."""
    if idx == 0 or idx == len(rid_sorted) - 1:
        return None, None
    rp, r, rn = rid_sorted[idx - 1], rid_sorted[idx], rid_sorted[idx + 1]
    Cp = atom(chain, rp, "C"); N = atom(chain, r, "N")
    CA = atom(chain, r, "CA"); C = atom(chain, r, "C"); Nn = atom(chain, rn, "N")
    if any(v is None for v in [Cp, N, CA, C, Nn]):
        return None, None
    return dihedral(Cp, N, CA, C), dihedral(N, CA, C, Nn)


# ── 로드 + 체인 식별 ──
nat = load(NATIVE_CIF)
s108 = load(S108_CIF)
nat_VH = nat[nat.chain_id == "B"]      # 1N8Z: B=heavy, A=light
nat_VL = nat[nat.chain_id == "A"]
s_chain = s108[s108.chain_id == "S"]   # s108: scFv 한 체인

DESIGN_CDR = {"H1": "TFSSYYMS", "H2": "IYSSGGSTYYAD", "H3": "GGRYGGYGFDY",
              "L1": "RASQSLSSSLY", "L2": "GASSRAS", "L3": "QQYNSPPLT"}

# 도메인별 canon→resid 매핑
nat_map = {"VH": map_canon_to_resid(VH, nat_VH), "VL": map_canon_to_resid(VL, nat_VL)}
s_map = {"VH": map_canon_to_resid(VH, s_chain), "VL": map_canon_to_resid(VL, s_chain)}
nat_chain = {"VH": nat_VH, "VL": nat_VL}
s_chain_d = {"VH": s_chain, "VL": s_chain}

# 매핑 sanity: 구조에서 뽑은 CDR 서열이 알려진 것과 맞나
print("=" * 78)
print(" 매핑 검증: 구조에서 추출한 CDR 서열")
print("=" * 78)
for cdr, (s, e, dom) in CDR.items():
    nseq = "".join(ProteinSequence.convert_letter_3to1(
        nat_chain[dom][(nat_chain[dom].res_id == nat_map[dom][p]) &
                       (nat_chain[dom].atom_name == "CA")].res_name[0])
        for p in range(s, e) if p in nat_map[dom])
    dseq = "".join(ProteinSequence.convert_letter_3to1(
        s_chain_d[dom][(s_chain_d[dom].res_id == s_map[dom][p]) &
                       (s_chain_d[dom].atom_name == "CA")].res_name[0])
        for p in range(s, e) if p in s_map[dom])
    print(f"  {cdr}: native={nseq:<13s} s108={dseq:<13s}  (기대 s108={DESIGN_CDR[cdr]})")

# ── 도메인별 framework Cα superpose ──
results = {}
overlay = {}   # cdr -> (nat_ca[N,3], s_ca[N,3]) aligned
for dom in ("VH", "VL"):
    cdr_pos = set()
    for c, (s, e, d) in CDR.items():
        if d == dom:
            cdr_pos |= set(range(s, e))
    common = [p for p in range(len(DOMSEQ[dom])) if p in nat_map[dom] and p in s_map[dom]]
    fw = [p for p in common if p not in cdr_pos]
    # framework Cα
    P = np.array([atom(s_chain_d[dom], s_map[dom][p], "CA") for p in fw])
    Q = np.array([atom(nat_chain[dom], nat_map[dom][p], "CA") for p in fw])
    R, t = kabsch(P, Q)
    fw_rmsd = np.sqrt(np.mean(np.sum((applyT(R, t, P) - Q) ** 2, 1)))
    results[dom] = {"fw_rmsd": fw_rmsd, "n_fw": len(fw), "R": R, "t": t}

    for c, (s, e, d) in CDR.items():
        if d != dom:
            continue
        pos = [p for p in range(s, e) if p in nat_map[dom] and p in s_map[dom]]
        # Cα
        sca = np.array([atom(s_chain_d[dom], s_map[dom][p], "CA") for p in pos])
        nca = np.array([atom(nat_chain[dom], nat_map[dom][p], "CA") for p in pos])
        sca_a = applyT(R, t, sca)
        ca_rmsd = np.sqrt(np.mean(np.sum((sca_a - nca) ** 2, 1)))
        overlay[c] = (nca, sca_a)
        # backbone N,CA,C,O
        bb_s, bb_n = [], []
        for p in pos:
            for nm in ("N", "CA", "C", "O"):
                a_s = atom(s_chain_d[dom], s_map[dom][p], nm)
                a_n = atom(nat_chain[dom], nat_map[dom][p], nm)
                if a_s is not None and a_n is not None:
                    bb_s.append(a_s); bb_n.append(a_n)
        bb_s, bb_n = applyT(R, t, np.array(bb_s)), np.array(bb_n)
        bb_rmsd = np.sqrt(np.mean(np.sum((bb_s - bb_n) ** 2, 1)))
        # side-chain (동일 AA 위치만)
        sc_s, sc_n, nmatch = [], [], 0
        for p in pos:
            rn_n = nat_chain[dom][(nat_chain[dom].res_id == nat_map[dom][p]) &
                                  (nat_chain[dom].atom_name == "CA")].res_name[0]
            rn_s = s_chain_d[dom][(s_chain_d[dom].res_id == s_map[dom][p]) &
                                  (s_chain_d[dom].atom_name == "CA")].res_name[0]
            if rn_n == rn_s:
                a_n = sidechain_atoms(nat_chain[dom], nat_map[dom][p])
                a_s = sidechain_atoms(s_chain_d[dom], s_map[dom][p])
                for nm in set(a_n) & set(a_s):
                    sc_n.append(a_n[nm]); sc_s.append(a_s[nm])
                nmatch += 1
        if sc_s:
            sc_s = applyT(R, t, np.array(sc_s)); sc_n = np.array(sc_n)
            sc_rmsd = np.sqrt(np.mean(np.sum((sc_s - sc_n) ** 2, 1)))
        else:
            sc_rmsd = None
        results[c] = {"ca_rmsd": ca_rmsd, "bb_rmsd": bb_rmsd,
                      "sc_rmsd": sc_rmsd, "n_id": nmatch, "n_res": len(pos)}

# ── 출력: RMSD 요약 ──
print("\n" + "=" * 78)
print(" CDR 구조 차이 — native(1N8Z) vs s108(ESMFold2), 도메인 framework 정렬")
print("=" * 78)
print(f"  framework Cα RMSD: VH {results['VH']['fw_rmsd']:.2f}Å ({results['VH']['n_fw']}res)"
      f" | VL {results['VL']['fw_rmsd']:.2f}Å ({results['VL']['n_fw']}res)  (정렬 품질)")
print(f"\n  {'CDR':4s}{'Cα RMSD':>10s}{'backbone RMSD':>15s}{'sidechain RMSD':>16s}{'(동일AA/총)':>12s}")
for c in ("H1", "H2", "H3", "L1", "L2", "L3"):
    r = results[c]
    sc = f"{r['sc_rmsd']:.2f}A" if r["sc_rmsd"] is not None else "  NA"
    ratio = f"{r['n_id']}/{r['n_res']}"
    print(f"  {c:4s}{r['ca_rmsd']:>9.2f}A{r['bb_rmsd']:>14.2f}A{sc:>16s}{ratio:>12s}")
print("\n  * side-chain RMSD = 동일 AA 위치만(설계로 잔기 바뀐 곳은 비교 불가). NA=동일 AA 없음.")

# ── phi/psi 테이블 ──
print("\n" + "=" * 78)
print(" CDR 잔기별 phi/psi (deg) — native vs s108")
print("=" * 78)
rama = {"nat": [], "s": []}
csv_lines = ["cdr,pos,native_aa,s108_aa,phi_nat,psi_nat,phi_s108,psi_s108,dphi,dpsi"]
for c in ("H1", "H2", "H3", "L1", "L2", "L3"):
    s, e, dom = CDR[c]
    nrids = sorted(nat_map[dom].values())
    srids = sorted(s_map[dom].values())
    print(f"\n[{c}]  {'pos':>4s} {'nat':>3s}{'s108':>5s} | {'phi_n':>7s}{'psi_n':>7s} | {'phi_s':>7s}{'psi_s':>7s} | {'Δphi':>6s}{'Δpsi':>6s}")
    for p in range(s, e):
        if p not in nat_map[dom] or p not in s_map[dom]:
            continue
        nr, sr = nat_map[dom][p], s_map[dom][p]
        na = ProteinSequence.convert_letter_3to1(
            nat_chain[dom][(nat_chain[dom].res_id == nr) & (nat_chain[dom].atom_name == "CA")].res_name[0])
        sa = ProteinSequence.convert_letter_3to1(
            s_chain_d[dom][(s_chain_d[dom].res_id == sr) & (s_chain_d[dom].atom_name == "CA")].res_name[0])
        phn, psn = phipsi(nat_chain[dom], nrids, nrids.index(nr))
        phs, pss = phipsi(s_chain_d[dom], srids, srids.index(sr))
        if None in (phn, psn, phs, pss):
            continue
        dphi = (phs - phn + 180) % 360 - 180
        dpsi = (pss - psn + 180) % 360 - 180
        print(f"      {p:>4d} {na:>3s}{sa:>5s} | {phn:>7.0f}{psn:>7.0f} | {phs:>7.0f}{pss:>7.0f} | {dphi:>6.0f}{dpsi:>6.0f}")
        rama["nat"].append((phn, psn)); rama["s"].append((phs, pss))
        csv_lines.append(f"{c},{p},{na},{sa},{phn:.1f},{psn:.1f},{phs:.1f},{pss:.1f},{dphi:.1f},{dpsi:.1f}")
open("report/cdr_phipsi_table.csv", "w").write("\n".join(csv_lines))
print("\nsaved: report/cdr_phipsi_table.csv")

# ── figure 1: 3D 오버레이 (도메인별) ──
CDR_COL = {"H1": "#e53935", "H2": "#fb8c00", "H3": "#8e24aa",
           "L1": "#1e88e5", "L2": "#00acc1", "L3": "#43a047"}
fig = plt.figure(figsize=(15, 7))
for si, dom in enumerate(("VH", "VL")):
    ax = fig.add_subplot(1, 2, si + 1, projection="3d")
    R, t = results[dom]["R"], results[dom]["t"]
    # framework Cα trace
    common = sorted(p for p in range(len(DOMSEQ[dom])) if p in nat_map[dom] and p in s_map[dom])
    nfw = np.array([atom(nat_chain[dom], nat_map[dom][p], "CA") for p in common])
    sfw = applyT(R, t, np.array([atom(s_chain_d[dom], s_map[dom][p], "CA") for p in common]))
    ax.plot(*nfw.T, color="#bbbbbb", lw=1.0, label="native framework")
    ax.plot(*sfw.T, color="#cfd8dc", lw=1.0, ls="--", label="s108 framework")
    for c, (s, e, d) in CDR.items():
        if d != dom:
            continue
        nca, sca = overlay[c]
        ax.plot(*nca.T, color=CDR_COL[c], lw=3, label=f"{c} native")
        ax.plot(*sca.T, color=CDR_COL[c], lw=3, ls=":", marker="o", ms=3, label=f"{c} s108")
    ax.set_title(f"{dom}  (framework-aligned; native=solid, s108=dotted)", fontsize=11)
    ax.legend(fontsize=6, loc="upper left")
    ax.set_axis_off()
plt.suptitle("CDR backbone overlay — native(1N8Z) vs s108(ESMFold2)", fontsize=13, weight="bold")
plt.tight_layout()
plt.savefig("report/fig_cdr_struct_overlay.png", dpi=150, bbox_inches="tight")
print("saved: report/fig_cdr_struct_overlay.png")

# ── figure 2: Ramachandran (표준 favored/allowed 배경) ──
def rama_background(ax):
    g = np.linspace(-180, 180, 220)
    P, Q = np.meshgrid(g, g)            # P=phi, Q=psi

    def blob(cx, cy, sx, sy, w):
        d = 0.0
        for dy in (-360, 0, 360):       # psi/phi 주기성
            for dx in (-360, 0, 360):
                d = d + w * np.exp(-(((P - cx - dx) / sx) ** 2 + ((Q - cy - dy) / sy) ** 2))
        return d
    dens = (blob(-63, -43, 22, 22, 1.0)     # right-handed alpha
            + blob(-120, 130, 38, 32, 1.0)  # beta-sheet
            + blob(-68, 150, 22, 22, 0.6)   # polyproline II
            + blob(60, 45, 20, 22, 0.32)    # left-handed alpha
            + blob(-120, -175, 40, 16, 0.22))  # beta (psi wrap)
    dens /= dens.max()
    # allowed(바깥, 연하게) + favored(안쪽, 진하게) 2단계 음영
    ax.contourf(P, Q, dens, levels=[0.0007, 0.025, 1.0],
                colors=["#e3eef7", "#b7d3ea"], zorder=0)
    ax.contour(P, Q, dens, levels=[0.0007, 0.025], colors="#7fa8cc", linewidths=0.7, zorder=1)


fig, ax = plt.subplots(figsize=(6.6, 6.6))
rama_background(ax)
ax.axhline(0, color="#cfcfcf", lw=0.6, zorder=1); ax.axvline(0, color="#cfcfcf", lw=0.6, zorder=1)
nat_a = np.array(rama["nat"]); s_a = np.array(rama["s"])
for (x0, y0), (x1, y1) in zip(nat_a, s_a):   # 같은 위치 연결(연하게)
    ax.plot([x0, x1], [y0, y1], color="#9e9e9e", lw=0.5, alpha=0.45, zorder=2)
ax.scatter(nat_a[:, 0], nat_a[:, 1], c="#6a3fb5", s=48, label="native", zorder=4,
           edgecolor="white", linewidth=0.7)
ax.scatter(s_a[:, 0], s_a[:, 1], c="#00897b", marker="^", s=52, label="s108", zorder=4,
           edgecolor="white", linewidth=0.7)
ax.set_xlim(-180, 180); ax.set_ylim(-180, 180); ax.set_aspect("equal")
ax.set_xticks(range(-180, 181, 90)); ax.set_yticks(range(-180, 181, 90))
ax.set_xlabel("phi (deg)", fontsize=11); ax.set_ylabel("psi (deg)", fontsize=11)
ax.set_title("Ramachandran — CDR residues: native vs s108\n"
             "(shaded = favored/allowed regions; grey line = same position)", fontsize=11)
ax.legend(loc="lower right", framealpha=0.95)
plt.tight_layout()
plt.savefig("report/fig_cdr_ramachandran.png", dpi=150, bbox_inches="tight")
print("saved: report/fig_cdr_ramachandran.png")

# ── PyMOL 스크립트 (인터랙티브) ──
pml = f"""# native(1N8Z) vs s108 CDR overlay
load {os.path.abspath(NATIVE_CIF)}, native
load {os.path.abspath(S108_CIF)}, s108
hide everything
# 항체 Fv 만 (native chain A/B, s108 chain S)
create nat_fv, native and chain A+B
create s108_fv, s108 and chain S
delete native
delete s108
align s108_fv, nat_fv
show cartoon, nat_fv or s108_fv
color grey70, nat_fv
color slate, s108_fv
set cartoon_transparency, 0.3, nat_fv
bg_color white
"""
open("report/align_s108_native.pml", "w").write(pml)
print("saved: report/align_s108_native.pml  (PyMOL: @report/align_s108_native.pml)")
