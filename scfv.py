"""scfv.py — 설계된 Fv(VH/VL) → scFv 단일사슬 포맷터.

scFv = VH–linker–VL (또는 VL–VH) 를 한 폴리펩타이드로 연결.
linker 표준: (G4S)3 = GGGGSGGGGSGGGGS (15-mer). 더 긴 (G4S)4 도 가능(monomeric 비율↑).
진행상황_260602.md §1 참조.

용도:
  - 최적 CDR graft 후 VH/VL 을 scFv 한 체인으로 병합 → ESMFold2 로 scFv+항원 폴딩.
  - 설계·최적화는 별도 체인(정확한 계면), 최종 산출물만 scFv 옵션으로.

linker 영역은 disorder 로 나올 가능성이 높으니 confidence 해석에 주의.
"""
from __future__ import annotations

import argparse

import yaml

LINKER_G4S3 = "GGGGSGGGGSGGGGS"        # (G4S)3, 15-mer (표준)
LINKER_G4S4 = "GGGGSGGGGSGGGGSGGGGS"   # (G4S)4, 20-mer


def make_scfv(vh_seq: str, vl_seq: str, linker: str = LINKER_G4S3,
              order: str = "VH-VL") -> dict:
    """VH/VL → scFv 단일사슬. 반환: {seq, linker_range(0-based [s,e)), order}."""
    vh = vh_seq.replace(" ", "").replace("\n", "")
    vl = vl_seq.replace(" ", "").replace("\n", "")
    if order.upper() == "VH-VL":
        seq = vh + linker + vl
        linker_range = (len(vh), len(vh) + len(linker))
    elif order.upper() == "VL-VH":
        seq = vl + linker + vh
        linker_range = (len(vl), len(vl) + len(linker))
    else:
        raise ValueError(f"order must be VH-VL or VL-VH, got {order}")
    return {"seq": seq, "linker_range": linker_range, "order": order.upper(),
            "length": len(seq)}


def write_scfv_yaml(path: str, scfv_seq: str, antigens: list[dict],
                    scfv_id: str = "S", num_loops: int = 10,
                    num_sampling_steps: int = 64) -> None:
    """scFv 단일사슬 + 항원(들) 폴딩용 YAML. antigens: [{'id','sequence'}, ...]."""
    chains = [{"type": "protein", "id": scfv_id, "sequence": scfv_seq}]
    chains += [{"type": "protein", "id": a["id"],
                "sequence": a["sequence"].replace(" ", "").replace("\n", "")}
               for a in antigens]
    doc = {"chains": chains,
           "inference": {"num_loops": num_loops, "num_sampling_steps": num_sampling_steps}}
    with open(path, "w") as f:
        yaml.dump(doc, f, sort_keys=False, allow_unicode=True)


def main():
    ap = argparse.ArgumentParser(description="Fv(VH/VL) → scFv 변환 + 폴딩 YAML 생성")
    ap.add_argument("--vh", required=True, help="VH 서열 (graft 완료)")
    ap.add_argument("--vl", required=True, help="VL 서열 (graft 완료)")
    ap.add_argument("--linker", default=LINKER_G4S3, help="linker 서열 (기본 (G4S)3)")
    ap.add_argument("--order", default="VH-VL", choices=["VH-VL", "VL-VH"])
    ap.add_argument("--antigen-id", default=None)
    ap.add_argument("--antigen-seq", default=None)
    ap.add_argument("--out-yaml", default=None, help="폴딩용 YAML 경로 (항원 제공 시)")
    args = ap.parse_args()

    sc = make_scfv(args.vh, args.vl, args.linker, args.order)
    print(f"scFv ({sc['order']}, {sc['length']} AA, linker[{sc['linker_range'][0]}:{sc['linker_range'][1]}]):")
    print(sc["seq"])

    if args.out_yaml and args.antigen_id and args.antigen_seq:
        write_scfv_yaml(args.out_yaml, sc["seq"],
                        [{"id": args.antigen_id, "sequence": args.antigen_seq}])
        print(f"\nfolding YAML → {args.out_yaml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
