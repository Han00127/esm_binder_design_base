"""esmc_prior.py — ESMC masked pseudo-PPL LM prior (Alg14, 충실 = soft 문맥).

논문대로 soft_binder 를 ESMC 입력으로 주입(soft 임베딩) → masked 위치 예측 → cross-entropy.
ESMCForMaskedLM 은 inputs_embeds 미지원 → esmc.embed 를 soft 임베딩 반환으로 *임시 패치*
(embedding lookup 우회, packing 로직은 그대로 재사용). gradient 가 문맥+라벨 둘 다로 흐름.

binder(20-AA, 우리 컬럼순서 ARNDCQEGHILKMFPSTWYV) → ESMC token id 매핑:
  ESMC_PERM[j] = j번째 우리컬럼 AA 의 ESMC token id.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM

ESMC_PERM = [5, 10, 17, 13, 23, 16, 9, 6, 21, 12, 4, 15, 20, 18, 14, 8, 11, 22, 19, 7]
OUR_AA = "ARNDCQEGHILKMFPSTWYV"     # 우리 컬럼 j 순서
CLS_ID, EOS_ID, MASK_ID = 0, 2, 32


class _ConstEmbed(torch.nn.Module):
    """esmc.embed 임시 대체: input_ids 무시하고 미리 만든 soft 임베딩 반환."""
    def __init__(self, val):
        super().__init__()
        self.val = val
    def forward(self, *a, **k):
        return self.val


class ESMCPrior:
    def __init__(self, path="/home/aidx/DB/weights/esmfold2/ESMC-600M", device="cuda"):
        self.m = AutoModelForMaskedLM.from_pretrained(path, local_files_only=True).to(device).eval()
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.esmc = self.m.esmc
        self.device = device
        self.perm = torch.tensor(ESMC_PERM, device=device)          # (20,) col→ESMC id
        self.emb = self.esmc.embed.weight                            # (64, d)

    def _forward_soft(self, x_emb):
        """soft 임베딩 (1,L+2,d) → lm_head logits (1,L+2,64). esmc.embed 패치."""
        orig = self.esmc.embed
        self.esmc.embed = _ConstEmbed(x_emb)                        # lookup 우회 (Module)
        try:
            L2 = x_emb.shape[1]
            dummy = torch.zeros(1, L2, dtype=torch.long, device=self.device)  # 0=cls, non-pad
            out = self.m(input_ids=dummy)
        finally:
            self.esmc.embed = orig
        return out.logits

    def score(self, soft_binder, masked_pos):
        """masked pseudo-PPL 1 pass. soft_binder (L,20) 우리순서, masked_pos = 가릴 위치 list.
        반환: -mean_{masked} Σ_a soft·log P_ESMC(a|soft문맥) (낮을수록 자연). 미분가능."""
        L = soft_binder.shape[0]
        aa_emb = self.emb[self.perm]                                # (20,d)
        x_aa = soft_binder @ aa_emb                                 # (L,d) soft 임베딩 (grad)
        if len(masked_pos) > 0:
            mp = torch.as_tensor(masked_pos, device=self.device)
            x_aa = x_aa.clone()
            x_aa[mp] = self.emb[MASK_ID]                            # 가린 위치 = mask 임베딩
        x = torch.cat([self.emb[CLS_ID][None], x_aa, self.emb[EOS_ID][None]], 0)[None]  # (1,L+2,d)
        logits = self._forward_soft(x)[0, 1:L + 1]                  # (L,64) AA 위치
        logp = F.log_softmax(logits.float(), -1)[:, self.perm]     # (L,20) 우리순서
        if len(masked_pos) == 0:
            return soft_binder.new_zeros(())
        mp = torch.as_tensor(masked_pos, device=self.device)
        return -(soft_binder[mp] * logp[mp]).sum(-1).mean()


def _seq_to_soft(seq, device):
    col = {a: j for j, a in enumerate(OUR_AA)}
    s = torch.zeros(len(seq), 20, device=device)
    for i, a in enumerate(seq):
        s[i, col.get(a, 0)] = 1.0
    return s


if __name__ == "__main__":
    import os, yaml
    dev = "cuda"
    prior = ESMCPrior(device=dev)
    print("[test] ESMC-600M prior 로드 OK")
    cfg = yaml.safe_load(open("configs/trastuzumab_her2.yaml"))
    vh = cfg["antibody"]["heavy"]["vh_sequence"].replace(" ", "").replace("\n", "")
    # 자연 H3 vs 방향족 쓰레기 H3 (smoke 산출물)
    nat = vh                                            # 자연 VH
    garbage = vh[:97] + "WWFWMYWFMWW" + vh[108:]        # H3(98-108) 자리에 쓰레기
    h3 = list(range(97, 108))                           # 가릴 위치(H3 근방)
    for name, seq in [("natural VH", nat), ("garbage-H3 VH", garbage)]:
        s = _seq_to_soft(seq, dev)
        with torch.no_grad():
            v = prior.score(s, h3)
        print(f"[test] {name:14s}: pseudo-PPL(H3) = {float(v):.4f}")
    # gradient 흐름
    s = _seq_to_soft(nat, dev).requires_grad_(True)
    v = prior.score(s, h3); v.backward()
    g = s.grad[h3].abs().mean().item()
    print(f"[test] gradient(mean@H3) = {g:.3e}  {'★흐름 OK' if g>0 else '✗0'}")
