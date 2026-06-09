"""composition.py — 자연 CDR 분포 타깃 손실 (composition-KL). 런타임(esmfold2): anarci 불필요.

build_pssm.py 가 만든 q_target.npz(위치별 q_pos[n,20] + q_global[20], AA 순서 'ARNDCQEGHILKMFPSTWYV')
를 로드 → 설계 j-순서(res_type id-2)로 재정렬 → L_comp = mean_pos KL(p_pos ‖ q_pos).

KL(p‖q)=Σ p·log(p/q): 설계 p 가 자연에서 드문 AA(W/F 과잉)에 질량 주면 커짐 → 자연 분포로 끌어당김.
(자연 수준 Tyr/방향족은 q 에 이미 있어 허용; 초과만 페널티.)
"""
from __future__ import annotations

import numpy as np
import torch

from esm.models.esmfold2.constants import PROTEIN_3TO1, PROTEIN_RESIDUE_TO_RES_TYPE

AA_BASE = 2


def _j_to_letter():
    """설계 컬럼 j(0..19) → AA 1글자 (res_type id = j+2)."""
    id2aa = {PROTEIN_RESIDUE_TO_RES_TYPE[k]: PROTEIN_3TO1[k] for k in PROTEIN_3TO1}
    return [id2aa[j + AA_BASE] for j in range(20)]


class CompositionTarget:
    """q_target.npz 로드 → 설계 j-순서 q_pos[n,20], q_global[20] 텐서 + KL 손실."""

    def __init__(self, npz_path, device, blend_global: float = 0.0):
        d = np.load(npz_path, allow_pickle=True)
        aa_order = str(d["aa_order"])
        q_pos = np.asarray(d["q_pos"], dtype=np.float64)          # [n,20] in aa_order
        q_global = np.asarray(d["q_global"], dtype=np.float64)    # [20]
        reidx = [aa_order.index(l) for l in _j_to_letter()]       # aa_order col for design col j
        qp = q_pos[:, reidx]
        qg = q_global[reidx]
        if blend_global > 0:                                      # 데이터 빈약 위치 robust화
            qp = (1 - blend_global) * qp + blend_global * qg[None, :]
        self.q_pos = torch.tensor(qp, device=device, dtype=torch.float32).clamp_min(1e-8)
        self.q_global = torch.tensor(qg, device=device, dtype=torch.float32).clamp_min(1e-8)
        self.n = qp.shape[0]
        self.support = np.asarray(d["support"]) if "support" in d else None
        self._arom_j = [i for i, l in enumerate(_j_to_letter()) if l in "WYF"]

    def kl(self, soft_cdr: torch.Tensor) -> torch.Tensor:
        """soft_cdr [n,20] (j-순서, cdr_idx 순서) → mean_pos KL(p‖q_pos)."""
        assert soft_cdr.shape[0] == self.n, f"CDR {soft_cdr.shape[0]} != target {self.n}"
        p = soft_cdr.clamp_min(1e-8)
        return (p * (p.log() - self.q_pos.log())).sum(-1).mean()

    def nll(self, soft_cdr: torch.Tensor) -> torch.Tensor:
        """Cross-Entropy(프로파일 NLL): L = mean_pos( -Σ_a p_a·log q_pos_a ).
        KL 과 달리 -H(p) 항이 없어 temperature annealing(p→one-hot)과 충돌 X → 폭발 X.
        설계 질량을 자연에서 흔한 AA 로 끎. (LM masked-PPL 과 동형.)"""
        assert soft_cdr.shape[0] == self.n, f"CDR {soft_cdr.shape[0]} != target {self.n}"
        return -(soft_cdr * self.q_pos.log()).sum(-1).mean()

    def arom_fraction(self, soft_cdr: torch.Tensor) -> float:
        """진단용: 설계 방향족(W+Y+F) 평균 분율."""
        return float(soft_cdr[:, self._arom_j].sum(-1).mean())

    def seq_nll(self, j_indices) -> float:
        """이산 서열(위치별 선택 AA 의 j-인덱스, cdr_idx 순서) → 위치별 자연성 NLL.
        mean_pos( -log q_pos[i, j_i] ). 낮을수록 자연 분포에 가까움. (native 와 비교용)."""
        idx = torch.tensor(list(j_indices), device=self.q_pos.device, dtype=torch.long)
        rows = torch.arange(self.n, device=idx.device)
        return float(-(self.q_pos[rows, idx].log()).mean())
