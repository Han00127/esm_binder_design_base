"""Ramachandran — framework(회색) 대비 CDR(region별 색). native + s108.
각 CDR(H1..L3)이 phi/psi 공간 어디에 앉는지 + frame과 어떻게 다른지.
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
plt.rcParams["axes.unicode_minus"] = False

VH = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
VL = "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK"
CDR = {"H1": (27, 35, "VH"), "H2": (50, 62, "VH"), "H3": (98, 109, "VH"),
       "L1": (23, 34, "VL"), "L2": (49, 56, "VL"), "L3": (88, 97, "VL")}
DOMSEQ = {"VH": VH, "VL": VL}
CDR_COL = {"H1": "#e53935", "H2": "#fb8c00", "H3": "#8e24aa",
           "L1": "#1e88e5", "L2": "#00acc1", "L3": "#43a047"}
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
    cseq, rids = chain_seq(chain)
    ali = align_optimal(ProteinSequence(canon), ProteinSequence(cseq.replace("X", "A")),
                        MATRIX, gap_penalty=(-10, -1), local=True)[0]
    return {a: int(rids[b]) for a, b in ali.trace if a >= 0 and b >= 0}


def atom(chain, rid, name):
    sel = chain[(chain.res_id == rid) & (chain.atom_name == name)]
    return sel.coord[0] if sel.array_length() > 0 else None


def dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))


def phipsi(chain, rids_sorted, idx):
    if idx == 0 or idx == len(rids_sorted) - 1:
        return None, None
    rp, r, rn = rids_sorted[idx - 1], rids_sorted[idx], rids_sorted[idx + 1]
    Cp, N, CA, C, Nn = (atom(chain, rp, "C"), atom(chain, r, "N"), atom(chain, r, "CA"),
                        atom(chain, r, "C"), atom(chain, rn, "N"))
    if any(v is None for v in (Cp, N, CA, C, Nn)):
        return None, None
    return dihedral(Cp, N, CA, C), dihedral(N, CA, C, Nn)


def region_of(dom, canon_pos):
    for c, (s, e, d) in CDR.items():
        if d == dom and s <= canon_pos < e:
            return c
    return "FR"


def collect(chain, dom):
    """V-domain 잔기별 (phi, psi, region). frame+CDR."""
    cmap = map_canon_to_resid(DOMSEQ[dom], chain)
    resid2region = {rid: region_of(dom, cp) for cp, rid in cmap.items()}
    rids_all = sorted(np.unique(chain.res_id).tolist())
    out = []
    for i, rid in enumerate(rids_all):
        if rid not in resid2region:           # V-domain 밖(상수영역) 제외
            continue
        ph, ps = phipsi(chain, rids_all, i)
        if ph is None:
            continue
        out.append((ph, ps, resid2region[rid]))
    return out


def rama_background(ax):
    g = np.linspace(-180, 180, 220)
    P, Q = np.meshgrid(g, g)

    def blob(cx, cy, sx, sy, w):
        d = 0.0
        for dy in (-360, 0, 360):
            for dx in (-360, 0, 360):
                d += w * np.exp(-(((P - cx - dx) / sx) ** 2 + ((Q - cy - dy) / sy) ** 2))
        return d
    dens = (blob(-63, -43, 22, 22, 1.0) + blob(-120, 130, 38, 32, 1.0)
            + blob(-68, 150, 22, 22, 0.6) + blob(60, 45, 20, 22, 0.32)
            + blob(-120, -175, 40, 16, 0.22))
    dens /= dens.max()
    ax.contourf(P, Q, dens, levels=[0.0007, 0.025, 1.0], colors=["#eef3f8", "#cfe0ee"], zorder=0)
    ax.contour(P, Q, dens, levels=[0.0007, 0.025], colors="#9bbcd8", linewidths=0.6, zorder=1)


STRUCTS = [("native (1N8Z)", "runs/rank_ce_out/1N8Z.cif", {"VH": "B", "VL": "A"}, None),
           ("s108 (ESMFold2)", "runs/rank_ce_out/s108_exp_full_2021.cif", {"VH": "S", "VL": "S"}, None)]

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
for ax, (label, path, chains, _) in zip(axes, STRUCTS):
    arr = load(path)
    pts = []
    for dom in ("VH", "VL"):
        pts += collect(arr[arr.chain_id == chains[dom]], dom)
    rama_background(ax)
    ax.axhline(0, color="#ddd", lw=0.6, zorder=1); ax.axvline(0, color="#ddd", lw=0.6, zorder=1)
    # framework = 회색 작은 점
    fr = [(p, q) for p, q, r in pts if r == "FR"]
    if fr:
        fr = np.array(fr)
        ax.scatter(fr[:, 0], fr[:, 1], s=14, c="#b0b0b0", alpha=0.55, zorder=2,
                   label=f"framework (n={len(fr)})", edgecolor="none")
    # CDR = region별 색
    for c in ("H1", "H2", "H3", "L1", "L2", "L3"):
        cc = [(p, q) for p, q, r in pts if r == c]
        if not cc:
            continue
        cc = np.array(cc)
        ax.scatter(cc[:, 0], cc[:, 1], s=55, c=CDR_COL[c], zorder=4,
                   edgecolor="white", linewidth=0.6, label=f"{c} (n={len(cc)})")
    ax.set_xlim(-180, 180); ax.set_ylim(-180, 180); ax.set_aspect("equal")
    ax.set_xticks(range(-180, 181, 90)); ax.set_yticks(range(-180, 181, 90))
    ax.set_xlabel("phi (deg)"); ax.set_ylabel("psi (deg)")
    ax.set_title(label, fontsize=12, weight="bold")
    ax.legend(loc="lower right", fontsize=7, framealpha=0.95, ncol=2)
fig.suptitle("Ramachandran by region — framework (grey) vs CDR (colored)",
             fontsize=14, weight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("report/fig_cdr_ramachandran_byregion.png", dpi=150, bbox_inches="tight")
print("saved: report/fig_cdr_ramachandran_byregion.png")
