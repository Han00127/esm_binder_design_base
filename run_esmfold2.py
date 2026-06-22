"""ESMFold2 YAML-driven inference script with auto-MSA pipeline.

Usage:
    python run_esmfold2.py --input <input.yaml> [--output <out.cif>] [--weights <path>]
                           [--msa {yaml,auto,none}]

MSA 모드:
    yaml (기본) : YAML의 msa 블록 사용 (이전 동작)
    auto        : sha256 해시 캐시 확인 → 없으면 colabfold_search 실행 후 캐시 저장
                  protein 체인 2개 이상이면 paired MSA (key=N taxonomy 태그) 자동 생성
    none        : MSA 없이 단일 서열 예측

환경변수:
    ESMFOLD2_WEIGHTS  : 기본 가중치 경로
    COLABFOLD_BIN     : colabfold_search 실행 파일 경로
    MMSEQS_BIN        : mmseqs 실행 파일 경로
    MSA_DB            : colabfold DB 디렉토리
    MSA_CACHE_DIR     : sha256 해시 기반 캐시 저장 경로

YAML 포맷은 1mht_input.yaml / 1brs_input.yaml 참조.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import string
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

from esm.utils.msa import MSA
from esm.utils.parsing import FastaEntry, read_sequences
from esm.models.esmfold2 import (
    DNAInput,
    ESMFold2InputBuilder,
    LigandInput,
    Modification,
    ProteinInput,
    RNAInput,
    StructurePredictionInput,
)
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model


# ── 경로 기본값 (환경변수로 덮어쓰기 가능) ───────────────────────────────────
DEFAULT_WEIGHTS = os.environ.get(
    "ESMFOLD2_WEIGHTS",
    "/home/aidx/DB/weights/esmfold2/ESMFold2",
)
COLABFOLD_BIN = os.environ.get(
    "COLABFOLD_BIN",
    "/home/kyeongtak/project/mother_model/boltz/localcolabfold/colabfold-conda/bin/colabfold_search",
)
MMSEQS_BIN = os.environ.get(
    "MMSEQS_BIN",
    "/home/kyeongtak/project/mother_model/boltz/localcolabfold/colabfold-conda/bin/mmseqs",
)
MSA_DB = os.environ.get(
    "MSA_DB",
    "/home/aidx/DB/MSA/boltz-mmseq",
)
MSA_CACHE_DIR = os.environ.get(
    "MSA_CACHE_DIR",
    "/home/kyeongtak/structure_projects/msa",
)
# auto 모드에서 greedy_select 기본값
MSA_AUTO_SELECT = 1024


def _parse_modifications(raw: list[dict] | None) -> list[Modification] | None:
    if not raw:
        return None
    return [Modification(position=m["position"], ccd=m["ccd"]) for m in raw]


_REMOVE_LOWER = str.maketrans(dict.fromkeys(string.ascii_lowercase))


def _load_msa_from_path(
    a3m_path: Path,
    max_seqs: int | None = None,
    select: int | None = None,
) -> MSA | None:
    """a3m 파일에서 MSA 객체를 로드 (길이 불일치 자동 제외 + greedy_select)."""
    if not a3m_path.exists():
        raise FileNotFoundError(f"MSA 파일을 찾을 수 없습니다: {a3m_path}")

    entries: list[FastaEntry] = []
    expected_len: int | None = None
    skipped = 0
    for header, seq in read_sequences(a3m_path):
        seq_clean = seq.translate(_REMOVE_LOWER)
        if expected_len is None:
            expected_len = len(seq_clean)
        if len(seq_clean) != expected_len:
            skipped += 1
            continue
        entries.append(FastaEntry(header, seq_clean))
        if max_seqs is not None and len(entries) >= max_seqs:
            break

    if skipped:
        print(f"  MSA: {skipped}개 서열 길이 불일치로 제외됨")

    msa = MSA(entries)
    if select is not None and msa.depth > select:
        msa = msa.greedy_select(num_seqs=select)

    print(f"  MSA loaded: depth={msa.depth}, seqlen={msa.seqlen} (from {a3m_path.name})")
    return msa


def _load_msa(msa_cfg: dict | None, yaml_dir: Path) -> MSA | None:
    """YAML의 msa 블록을 읽어 MSA 객체 반환 (yaml 모드)."""
    if msa_cfg is None:
        return None
    a3m = msa_cfg.get("a3m")
    if a3m is None:
        return None
    a3m_path = Path(a3m)
    if not a3m_path.is_absolute():
        a3m_path = yaml_dir / a3m_path
    return _load_msa_from_path(
        a3m_path,
        max_seqs=msa_cfg.get("max_sequences"),
        select=msa_cfg.get("select"),
    )


def _get_or_create_msa(sequence: str, select: int = MSA_AUTO_SELECT) -> MSA | None:
    """sha256 해시 캐시 확인 → 없으면 colabfold_search 실행 → 캐시 저장 → MSA 반환.

    캐시 디렉토리: MSA_CACHE_DIR (=/home/kyeongtak/structure_projects/msa)
    """
    seq_hash = hashlib.sha256(sequence.encode()).hexdigest()
    cache_dir = Path(MSA_CACHE_DIR)
    cached_a3m = cache_dir / f"{seq_hash}.a3m"

    if cached_a3m.exists() and cached_a3m.stat().st_size > 0:
        print(f"  [MSA] cache hit  → {cached_a3m.name}")
        return _load_msa_from_path(cached_a3m, select=select)

    # ── cache miss: colabfold_search 실행 ────────────────────────────────
    print(f"  [MSA] cache miss (hash: {seq_hash[:16]}...)")
    print(f"  [MSA] colabfold_search 실행 중 (DB: {MSA_DB}) ...")

    # 실행 파일 확인
    for exe, name in [(COLABFOLD_BIN, "colabfold_search"), (MMSEQS_BIN, "mmseqs")]:
        if not Path(exe).exists():
            raise FileNotFoundError(
                f"{name} 실행 파일을 찾을 수 없습니다: {exe}\n"
                f"환경변수 {name.upper().replace('_SEARCH','').replace('COLABFOLD','COLABFOLD_BIN')} 로 경로를 지정하세요."
            )

    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="esmfold2_msa_") as tmpdir:
        tmpdir = Path(tmpdir)
        fasta_path = tmpdir / "query.fasta"
        fasta_path.write_text(f">query\n{sequence}\n")

        out_dir = tmpdir / "msa_out"
        out_dir.mkdir()

        cmd = [
            COLABFOLD_BIN,
            "--mmseqs", MMSEQS_BIN,
            "--db1", "uniref30_2302_db",
            "--db3", "colabfold_envdb_202108_db",
            "--db-load-mode", "2",   # mmap: NFS 대신 로컬 NVMe .idx 사용 (~10x 속도)
            str(fasta_path),
            MSA_DB,
            str(out_dir),
        ]
        result = subprocess.run(
            cmd,
            capture_output=False,  # 진행 상황 출력
            check=True,
        )

        elapsed = time.time() - t0
        print(f"  [MSA] 검색 완료: {elapsed:.1f}초 ({elapsed/60:.1f}분)")

        # colabfold_search 출력 파일 찾기 (query.a3m 또는 0.a3m)
        a3m_candidates = list(out_dir.glob("*.a3m"))
        if not a3m_candidates:
            raise RuntimeError(f"colabfold_search 출력에서 .a3m 파일을 찾을 수 없습니다: {out_dir}")
        result_a3m = a3m_candidates[0]

        # 캐시 디렉토리 생성 후 복사
        cache_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(result_a3m, cached_a3m)
        print(f"  [MSA] 캐시 저장 → {cached_a3m}")

    return _load_msa_from_path(cached_a3m, select=select)


# ── paired MSA 헬퍼 (복합체 auto 모드) ───────────────────────────────────────

def _parse_a3m(text: str) -> list[tuple[str, str]]:
    """a3m 텍스트 → [(header, sequence), ...]"""
    entries: list[tuple[str, str]] = []
    header: str | None = None
    buf: list[str] = []
    for line in text.splitlines():
        if line.startswith(">"):
            if header is not None:
                entries.append((header, "".join(buf)))
            header = line[1:]
            buf = []
        elif line.strip():
            buf.append(line.strip())
    if header is not None:
        entries.append((header, "".join(buf)))
    return entries


def _split_match_cols(seq: str, lengths: list[int]) -> list[str]:
    """combined a3m 서열을 체인별로 분리 (match col = 대문자 or '-')."""
    parts: list[str] = []
    pos = 0
    n = len(seq)
    for length in lengths:
        n_match = 0
        start = pos
        while pos < n:
            ch = seq[pos]
            if ch.isupper() or ch == "-":
                n_match += 1
                pos += 1
                if n_match == length:
                    break
            else:
                pos += 1
        parts.append(seq[start:pos])
    return parts


def _has_residues(seq: str) -> bool:
    return any(c.isupper() for c in seq)


def _extract_chain_header(combined_header: str, chain_idx: int) -> str:
    """tab 구분 combined 헤더에서 chain_idx 번째 헤더 추출."""
    parts = combined_header.split("\t")
    if chain_idx < len(parts):
        return parts[chain_idx].lstrip(">").strip()
    return combined_header.strip()


def _split_combined_a3m(
    a3m_text: str,
    chain_lengths: list[int],
) -> tuple[list[list[tuple[str, str]]], int]:
    """block-diagonal combined a3m → 체인별 (header, seq) 리스트 + key=N 태그.

    Returns:
        per_chain : per_chain[i] = [(header_with_key, seq), ...]
        n_paired  : paired 행 수 (query 제외)
    """
    entries = _parse_a3m(a3m_text)
    if not entries:
        raise ValueError("combined a3m 파일이 비어있습니다")

    n_chains = len(chain_lengths)
    per_chain: list[list[tuple[str, str]]] = [[] for _ in range(n_chains)]
    n_paired = 0
    paired_key = 1

    # Row 0: query (ESMFold2가 항상 paired로 처리, key= 불필요)
    q_header, q_seq = entries[0]
    q_parts = _split_match_cols(q_seq, chain_lengths)
    for ci in range(n_chains):
        per_chain[ci].append((_extract_chain_header(q_header, ci), q_parts[ci]))

    for header, seq in entries[1:]:
        parts = _split_match_cols(seq, chain_lengths)
        present = [_has_residues(p) for p in parts]
        if not any(present):
            continue
        if all(present):
            key = paired_key
            paired_key += 1
            n_paired += 1
        else:
            key = -1
        for ci in range(n_chains):
            if present[ci]:
                h = _extract_chain_header(header, ci)
                per_chain[ci].append((f"{h} key={key}", parts[ci]))

    return per_chain, n_paired


def _write_a3m(entries: list[tuple[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for header, seq in entries:
            f.write(f">{header}\n{seq}\n")


def _get_or_create_paired_msas(
    protein_chains: list[dict],
    yaml_dir: Path,
    select: int = MSA_AUTO_SELECT,
) -> dict[str, MSA]:
    """복합체용 paired MSA 생성/캐시 → {chain_id: MSA} 반환.

    캐시 구조 (MSA_CACHE_DIR 기준):
      combined a3m : {sha256(SEQ_A:SEQ_B)}.a3m           ← colabfold_search 원본
      per-chain    : {sha256(SEQ_A:SEQ_B)}_{chain_id}.a3m ← 체인별 분리 결과

    hit 판정 (3단계):
      1. per-chain 파일 전부 존재  → 완전 hit (검색+분리 skip)
      2. combined만 존재           → combined hit (검색 skip, 분리만 실행)
      3. 둘 다 없음                → full miss (검색 → combined 저장 → 분리)
    """
    sequences = [c["sequence"].replace("\n", "").replace(" ", "") for c in protein_chains]
    chain_ids = [c["id"] for c in protein_chains]
    chain_lengths = [len(s) for s in sequences]

    # sha256 해시: 복합체 전체 (결합 서열)
    combined_seq = ":".join(sequences)
    combined_hash = hashlib.sha256(combined_seq.encode()).hexdigest()
    cache_dir = Path(MSA_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cached_combined = cache_dir / f"{combined_hash}.a3m"
    # per-chain 캐시 경로: MSA_CACHE_DIR/{combined_hash}_{chain_id}.a3m
    per_chain_paths: list[Path] = [
        cache_dir / f"{combined_hash}_{cid}.a3m" for cid in chain_ids
    ]

    # ── 1단계: per-chain 파일 전부 hit → 검색+분리 완전 skip ────────────────
    if all(p.exists() and p.stat().st_size > 0 for p in per_chain_paths):
        print(f"  [paired MSA] cache hit (hash: {combined_hash[:16]}...)")
        result_msas: dict[str, MSA] = {}
        for cid, p in zip(chain_ids, per_chain_paths):
            result_msas[cid] = _load_msa_from_path(p, select=select)
        return result_msas

    # ── 2단계: combined a3m 캐시 확인 / 없으면 colabfold_search 실행 ────────
    if not (cached_combined.exists() and cached_combined.stat().st_size > 0):
        print(f"  [paired MSA] cache miss (hash: {combined_hash[:16]}...)")
        print(f"  [paired MSA] {len(sequences)}개 체인 복합체 MSA 검색 중 ...")

        for exe, name in [(COLABFOLD_BIN, "colabfold_search"), (MMSEQS_BIN, "mmseqs")]:
            if not Path(exe).exists():
                raise FileNotFoundError(f"{name} 실행 파일을 찾을 수 없습니다: {exe}")

        t0 = time.time()
        with tempfile.TemporaryDirectory(prefix="esmfold2_paired_") as tmpdir:
            tmpdir_path = Path(tmpdir)
            fasta_path = tmpdir_path / "query.fasta"
            fasta_path.write_text(f">query\n{':'.join(sequences)}\n")
            out_dir = tmpdir_path / "cf_out"
            out_dir.mkdir()
            cmd = [
                COLABFOLD_BIN, "--mmseqs", MMSEQS_BIN,
                "--db1", "uniref30_2302_db",
                "--db3", "colabfold_envdb_202108_db",
                "--db-load-mode", "2",
                str(fasta_path), MSA_DB, str(out_dir),
            ]
            subprocess.run(cmd, check=True)
            elapsed = time.time() - t0
            print(f"  [paired MSA] 검색 완료: {elapsed:.1f}초")

            a3m_files = sorted(out_dir.glob("*.a3m"))
            if not a3m_files:
                raise RuntimeError(f"colabfold_search 출력에서 .a3m 파일을 찾을 수 없습니다: {out_dir}")
            result_a3m = max(a3m_files, key=lambda p: p.stat().st_size)
            shutil.copy2(result_a3m, cached_combined)
            print(f"  [paired MSA] combined 캐시 저장 → {cached_combined.name}")
    else:
        print(f"  [paired MSA] combined hit, per-chain 분리 중 (hash: {combined_hash[:16]}...)")

    # ── 3단계: combined a3m → 체인별 분리 + key=N 태그 → per-chain 캐시 저장 ─
    a3m_text = cached_combined.read_text()
    per_chain, n_paired = _split_combined_a3m(a3m_text, chain_lengths)
    total_entries = sum(len(p) for p in per_chain)
    print(f"  [paired MSA] 총 {total_entries}개 서열 분리 완료, paired={n_paired}")

    result_msas = {}
    for ci, (cid, cache_path) in enumerate(zip(chain_ids, per_chain_paths)):
        entries = per_chain[ci]
        n_unpaired = sum(1 for h, _ in entries if "key=-1" in h)
        _write_a3m(entries, cache_path)
        print(f"  [Chain {cid}] total={len(entries)}, paired={n_paired}, unpaired={n_unpaired} → {cache_path.name}")
        result_msas[cid] = _load_msa_from_path(cache_path, select=select)

    return result_msas


def _build_spi(
    chains: list[dict],
    yaml_dir: Path,
    msa_mode: str = "yaml",
    select: int = MSA_AUTO_SELECT,
) -> StructurePredictionInput:
    """msa_mode: 'yaml' | 'auto' | 'none'

    auto 모드 분기:
      protein 체인 2개 이상 → paired MSA (_get_or_create_paired_msas)
      protein 체인 1개       → 단일 서열 MSA (_get_or_create_msa, sha256 캐시)
    """
    # auto 모드 + 복합체: protein 체인 목록을 먼저 수집하여 paired MSA 한 번에 처리
    protein_chains_data = [c for c in chains if c["type"].lower() == "protein"]
    paired_msas: dict[str, MSA] = {}
    if msa_mode == "auto" and len(protein_chains_data) >= 2:
        # ── 동일서열(homo-oligomer) dedup → unique 만 검색 후 broadcast ──
        # colabfold_search 가 동일 체인을 dedup 해서 combined a3m 세그먼트 수가 줄면,
        # _split_combined_a3m 이 원래 체인수 길이로 자르다 어긋나 일부 체인이 빈 MSA →
        # construct_paired_msa 크래시. unique 로만 검색하고 같은 서열 체인끼리 MSA 공유해 회피.
        _norm = lambda s: s.replace("\n", "").replace(" ", "")
        uniq_chains: list[dict] = []
        seq2uid: dict[str, str] = {}
        for c in protein_chains_data:
            s = _norm(c["sequence"])
            if s not in seq2uid:
                seq2uid[s] = c["id"]; uniq_chains.append(c)
        if len(uniq_chains) < len(protein_chains_data):
            print(f"  [auto] 동일서열 dedup: {len(protein_chains_data)}체인 → "
                  f"unique {len(uniq_chains)}개 {[c['id'] for c in uniq_chains]} 검색 후 broadcast")
        print(f"  [auto] {len(uniq_chains)}개 unique protein 체인 → paired MSA 검색")
        uniq_msas = _get_or_create_paired_msas(uniq_chains, yaml_dir, select=select)
        paired_msas = {c["id"]: uniq_msas[seq2uid[_norm(c["sequence"])]]
                       for c in protein_chains_data}     # 같은 서열 체인 = 같은 MSA

    sequences = []
    for chain in chains:
        t = chain["type"].lower()
        cid = chain["id"]
        mods = _parse_modifications(chain.get("modifications"))

        if t == "protein":
            seq = chain["sequence"].replace("\n", "").replace(" ", "")
            if msa_mode == "none":
                msa = None
            elif msa_mode == "auto":
                if paired_msas:
                    # 복합체: paired MSA 결과 사용
                    msa = paired_msas[cid]
                else:
                    # 단일 체인: 기존 단일 서열 검색
                    print(f"  [Chain {cid}] auto-MSA 검색 중...")
                    msa = _get_or_create_msa(seq, select=select)
            else:  # yaml
                msa = _load_msa(chain.get("msa"), yaml_dir)
            sequences.append(ProteinInput(id=cid, sequence=seq, modifications=mods, msa=msa))

        elif t == "dna":
            seq = chain["sequence"].replace("\n", "").replace(" ", "")
            sequences.append(DNAInput(id=cid, sequence=seq, modifications=mods))

        elif t == "rna":
            seq = chain["sequence"].replace("\n", "").replace(" ", "")
            sequences.append(RNAInput(id=cid, sequence=seq, modifications=mods))

        elif t == "ligand":
            ccd = chain.get("ccd")
            smiles = chain.get("smiles")
            if ccd is None and smiles is None:
                raise ValueError(f"Ligand chain '{cid}' needs 'ccd' or 'smiles'.")
            sequences.append(LigandInput(id=cid, ccd=ccd, smiles=smiles))

        else:
            raise ValueError(f"Unknown chain type '{t}'. Use: protein | dna | rna | ligand")

    return StructurePredictionInput(sequences=sequences)


def main():
    parser = argparse.ArgumentParser(
        description="ESMFold2 YAML inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""MSA 모드:
  yaml (기본) : YAML의 msa 블록 사용
  auto        : sha256 해시 캐시 확인 → 없으면 colabfold_search 실행
  none        : MSA 없이 단일 서열 예측""",
    )
    parser.add_argument("--input", "-i", required=True, help="입력 YAML 파일 경로")
    parser.add_argument("--output", "-o", default=None, help="출력 CIF 파일 경로 (YAML의 output.cif 덮어쓰기)")
    parser.add_argument("--weights", "-w", default=DEFAULT_WEIGHTS, help="ESMFold2 가중치 경로")
    parser.add_argument(
        "--msa",
        choices=["yaml", "auto", "none"],
        default="yaml",
        metavar="{yaml,auto,none}",
        help="MSA 모드: yaml(기본) | auto(해시캐시+colabfold_search) | none(MSA 미사용)",
    )
    parser.add_argument(
        "--msa-select",
        type=int,
        default=MSA_AUTO_SELECT,
        metavar="N",
        help=f"auto 모드에서 greedy_select 최대 서열 수 (기본: {MSA_AUTO_SELECT})",
    )
    args = parser.parse_args()

    # ── YAML 로드 ──────────────────────────────────────────────────────────
    yaml_path = Path(args.input)
    if not yaml_path.exists():
        print(f"[ERROR] YAML 파일을 찾을 수 없습니다: {yaml_path}", file=sys.stderr)
        sys.exit(1)

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    # ── 입력 구성 ──────────────────────────────────────────────────────────
    chains = cfg.get("chains", [])
    if not chains:
        print("[ERROR] YAML에 'chains' 항목이 없습니다.", file=sys.stderr)
        sys.exit(1)

    print(f"MSA 모드: {args.msa}" + (f" (select={args.msa_select})" if args.msa != "yaml" else ""))
    spi = _build_spi(chains, yaml_path.parent, msa_mode=args.msa, select=args.msa_select)

    # ── 추론 파라미터 ──────────────────────────────────────────────────────
    infer = cfg.get("inference", {})
    num_loops             = infer.get("num_loops", 3)
    num_sampling_steps    = infer.get("num_sampling_steps", 200)
    num_diffusion_samples = infer.get("num_diffusion_samples", 1)
    seed                  = infer.get("seed", None)
    if isinstance(seed, list):
        seed = seed[0]
    if seed is not None:
        seed = int(seed)

    # ── 출력 경로 ─────────────────────────────────────────────────────────
    out_cfg = cfg.get("output", {})
    cif_path = args.output or out_cfg.get("cif", "output.cif")

    # ── 모델 로드 ─────────────────────────────────────────────────────────
    weights_path = args.weights
    print(f"Loading ESMFold2 model from: {weights_path}")
    model = ESMFold2Model.from_pretrained(
        weights_path,
        local_files_only=True,
    ).cuda().eval()
    print("Model loaded.")

    # ── 추론 실행 ─────────────────────────────────────────────────────────
    print(f"Running structure prediction (loops={num_loops}, steps={num_sampling_steps}, "
          f"diffusion_samples={num_diffusion_samples}, seed={seed})...")

    raw = ESMFold2InputBuilder().fold(
        model,
        spi,
        num_loops=num_loops,
        num_sampling_steps=num_sampling_steps,
        num_diffusion_samples=num_diffusion_samples,
        seed=seed,
    )

    # num_diffusion_samples > 1 이면 list, 1이면 단일 객체
    results = raw if isinstance(raw, list) else [raw]
    multi_sample = len(results) > 1

    base_out = Path(cif_path)
    base_out.parent.mkdir(parents=True, exist_ok=True)

    def _save_result(result, out: Path) -> None:
        """단일 MolecularComplexResult를 CIF + confidence + PAE + distogram 으로 저장."""
        plddt_per_token = result.plddt.float().cpu().numpy()
        plddt_mean = float(plddt_per_token.mean())
        ptm   = float(result.ptm)  if result.ptm  is not None else None
        iptm  = float(result.iptm) if result.iptm is not None else None
        score_line = f"pLDDT mean: {plddt_mean:.3f}"
        if ptm  is not None: score_line += f", pTM: {ptm:.3f}"
        if iptm is not None: score_line += f", ipTM: {iptm:.3f}"
        print(score_line)

        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            f.write(result.complex.to_mmcif())
        print(f"Saved: {out}")

        stem = out.with_suffix("")
        conf: dict = {
            "plddt_mean": round(plddt_mean, 4),
            "ptm":        round(ptm, 4)  if ptm  is not None else None,
            "iptm":       round(iptm, 4) if iptm is not None else None,
            "plddt_per_token": [round(float(v), 4) for v in plddt_per_token],
        }
        if result.pair_chains_iptm is not None:
            conf["pair_chains_iptm"] = result.pair_chains_iptm.float().cpu().numpy().tolist()
        conf_path = Path(str(stem) + "_confidence.json")
        conf_path.write_text(json.dumps(conf, indent=2))
        print(f"Saved: {conf_path}")

        if result.pae is not None:
            pae_np = result.pae.float().cpu().numpy()
            pae_path = Path(str(stem) + "_pae.npy")
            np.save(str(pae_path), pae_np)
            print(f"Saved: {pae_path}  (PAE {pae_np.shape}, mean={pae_np.mean():.2f} Å)")

        if result.distogram is not None:
            disto_np = result.distogram.float().cpu().numpy()
            disto_path = Path(str(stem) + "_distogram.npy")
            np.save(str(disto_path), disto_np)
            print(f"Saved: {disto_path}  (distogram {disto_np.shape})")

        return plddt_mean

    # ── 결과 저장 ─────────────────────────────────────────────────────────
    best_idx, best_plddt = 0, -1.0
    for i, result in enumerate(results):
        if multi_sample:
            out = base_out.with_name(base_out.stem + f"_sample{i}" + base_out.suffix)
            print(f"\n[sample {i}/{len(results)-1}]")
        else:
            out = base_out
        plddt_mean = _save_result(result, out)
        if plddt_mean > best_plddt:
            best_plddt, best_idx = plddt_mean, i

    if multi_sample:
        print(f"\n최고 pLDDT: sample {best_idx} ({best_plddt:.3f})")


if __name__ == "__main__":
    main()
