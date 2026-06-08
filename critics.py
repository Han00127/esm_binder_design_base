"""critics.py — ranking ensemble용 ESMFold2 Experimental critic 레지스트리.

논문(Algorithm 11 주변)의 low-compute ranking ensemble = 4개 experimental critic.
각 critic 으로 후보의 target-bound 복합체를 폴딩 → interface confidence(ipTM/ipSAE) +
distogram-proxy 를 매겨 평균 → 후보 ranking. (gradient 아님, forward-only 선택용.)

세팅 메모:
  · 위치: /home/aidx/DB/weights/esmfold2/esmfold2_critics/<name>/
  · esmc_id 는 로컬 ESMC-6B(/home/aidx/DB/weights/esmfold2/ESMC-6B)로 수정됨 (6B 공유).
  · ccd.pkl 은 base biohub/ESMFold2 HF 캐시에서 공유 로드.
  · ★ 아키텍처가 base ESMFold2와 다름: lm_encoder/parcae_coda 없음
    → model_hooks 의 confidence-grad 배선(parcae_coda hook) 적용 불가(랭킹엔 불필요).
"""
from __future__ import annotations

CRITIC_DIR = "/home/aidx/DB/weights/esmfold2/esmfold2_critics"

# key -> (repo 디렉터리명, 설명)
CRITICS = {
    "exp_full_2021": ("ESMFold2-Experimental",              "full, cutoff2021"),
    "exp_full_2025": ("ESMFold2-Experimental-Cutoff2025",   "full, cutoff2025"),
    "exp_fast_2021": ("ESMFold2-Experimental-Fast",         "fast, cutoff2021"),
    "exp_fast_2025": ("ESMFold2-Experimental-Fast-Cutoff2025", "fast, cutoff2025"),
}


def critic_path(key: str) -> str:
    name, _ = CRITICS[key]
    return f"{CRITIC_DIR}/{name}"


def all_critic_paths() -> dict[str, str]:
    return {k: critic_path(k) for k in CRITICS}


if __name__ == "__main__":
    import os
    for k, (name, desc) in CRITICS.items():
        p = critic_path(k)
        ok = os.path.isfile(f"{p}/config.json") and os.path.isfile(f"{p}/model.safetensors")
        print(f"  {k:14s} {'OK ' if ok else 'MISSING'} {desc:18s} {p}")
