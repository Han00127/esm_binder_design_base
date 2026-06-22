"""model_hooks.py — ESMFold2 를 gradient 설계용으로 배선.

해결하는 두 가지 차단 요인:
  (1) `@torch.inference_mode()` 데코레이터  → `forward.__wrapped__` 직접 호출.
  (2) confidence head 입력이 forward 에서 전부 `.detach()` (modeling_esmfold2.py:1019-1032)
      → `pae_logits` 에 grad 가 안 흐름.  우회:
        - parcae_coda 출력(z, line 985)을 forward hook 으로 **non-detached 캡처**
        - confidence_head 를 래퍼로 교체해, 넘어온 detached z 대신 캡처한 non-detached z 사용
      → `pae_logits` 가 z(=soft sequence) 까지 미분 가능. (x_pred 좌표는 detached=geometry 고정,
        gradient 는 pair representation 경로로만 흐름 — 합리적 근사.)

메모리: confidence head(folding_trunk) 를 grad 로 돌리므로 no-op 대비 ~4-5GB 추가. A100 80GB 면 OK.
diff-ipSAE/iptm 이 필요 없으면 `enable_confidence_grad=False` 로 두면 저렴(distogram-only).
"""
from __future__ import annotations

import types

import torch


def load_design_model(weights: str, device: str = "cuda",
                      enable_confidence_grad: bool = False):
    """설계용 ESMFold2 로드. 반환: (model, raw_forward_callable).

    enable_confidence_grad=False(기본, 검증됨): confidence head no-op → distogram-only gradient.
      인터페이스 신호는 losses.interface_tm_distogram(distogram 기반) 으로.
    enable_confidence_grad=True(실험적, backward 불안정): pae_logits 를 grad 로 노출하나
      confidence head 내부 inplace 연산(fused residual/checkpoint 외 추가)으로 backward 실패.
      → 현재 PAE 기반 미분 ipSAE 는 미지원. 진짜 ipSAE 는 filters.py(사후)에서 사용."""
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

    model = ESMFold2Model.from_pretrained(weights, local_files_only=True).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    _fwd = type(model).forward
    raw_fwd = getattr(_fwd, "__wrapped__", _fwd)        # inference_mode 우회(있으면); Experimental 변종은 미데코 → 직접

    if enable_confidence_grad:
        _wire_confidence_grad(model, device)
    else:
        # 저렴 모드: confidence head no-op (pae_logits 없음, distogram-only gradient).
        # 단, 저온 real ipTM 추적(Alg11 line12-15)을 위해 원본 confidence head 를 보관 →
        # esmfold_diff.iptm_confidence 에서 no_grad fold 시 일시 복원(swap-in).
        model._real_confidence_head = model.confidence_head

        class _NoOp(torch.nn.Module):
            def forward(self, *a, **kw):
                return {}
        model.confidence_head = _NoOp().to(device)

    return model, raw_fwd


def _wire_confidence_grad(model, device):
    """parcae_coda 출력 z 를 캡처해 confidence head 가 non-detached z 를 쓰도록 교체."""
    cap: dict[str, torch.Tensor] = {}

    def coda_hook(_module, _inp, out):
        cap["z"] = out                                 # non-detached (grad 보존)

    model.parcae_coda.register_forward_hook(coda_hook)

    # confidence head 는 원래 backprop 대상이 아니라 autograd 와 충돌하는 연산이 둘 있음:
    #  (a) fused DropoutResidual = inplace `residual.add_` → set_kernel_backend(None) 로 unfused(out-of-place)
    #  (b) FoldingTrunk 가 grad 시 block 마다 checkpoint(use_reentrant=False) → 내부 inplace 와 충돌
    # 둘 다 제거: unfused + checkpoint 해제.
    model.confidence_head.folding_trunk.set_kernel_backend(None)
    #  (c) ConfidenceHead.forward 의 `pair.add_(pair_delta.float())` (modeling_esmfold2.py:223,
    #      추론용 inplace residual) → out-of-place 로 치환해야 pae_logits backward 가능.
    _patch_confidence_forward_inplace(model.confidence_head)
    #  inplace 를 고쳤으니 folding_trunk checkpointing 을 켜둔 채로(default) backward 가능 →
    #  메모리 절약. (만약 backward 가 깨지면 아래 _disable_trunk_checkpointing 호출로 폴백.)
    # _disable_trunk_checkpointing(model.confidence_head.folding_trunk)

    orig_conf = model.confidence_head

    class _ConfGradWrap(torch.nn.Module):
        """forward 가 z.detach() 를 넘겨도, 캡처한 non-detached z 로 치환해 재계산."""
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, *, z, **kw):
            z_grad = cap.get("z", None)
            if z_grad is not None:
                # shape 일치 확인 후 non-detached z 로 치환 (dtype 은 inner 가 처리)
                if z_grad.shape == z.shape:
                    z = z_grad.float()
            return self.inner(z=z, **kw)

    model.confidence_head = _ConfGradWrap(orig_conf).to(device)


def _patch_confidence_forward_inplace(conf_head):
    """ConfidenceHead.forward 의 inplace `pair.add_(pair_delta.float())` → out-of-place.
    소스를 가져와 한 줄만 치환 후 모듈 namespace 에서 재정의 (다른 참조 이름 보존)."""
    import inspect, sys, textwrap
    cls = type(conf_head)
    src = textwrap.dedent(inspect.getsource(cls.forward))
    needle = "pair.add_(pair_delta.float())"
    if needle not in src:
        print(f"[model_hooks] WARN: '{needle}' 못 찾음 (코드 버전 변경?) — 패치 skip")
        return False
    src = src.replace(needle, "pair = pair + pair_delta.float()")
    ns = dict(vars(sys.modules[cls.__module__]))   # 모듈 전역(참조 이름) 확보
    exec(src, ns)
    cls.forward = ns["forward"]                     # 클래스 메서드 교체(프로세스 한정)
    return True


def _disable_trunk_checkpointing(folding_trunk):
    """FoldingTrunk.forward 를 checkpoint 없이 블록 순차 실행하도록 인스턴스 패치."""
    from transformers.models.esmfold2.modeling_esmfold2_common import BACKEND_FUSED

    def _no_ckpt_forward(self, pair, pair_attention_mask=None):
        orig_dtype = pair.dtype
        fused_on = (len(self.blocks) > 0
                    and getattr(self.blocks[0], "_kernel_backend", None) == BACKEND_FUSED)
        if pair.is_cuda and fused_on and orig_dtype != torch.bfloat16:
            pair = pair.to(torch.bfloat16)
        for block in self.blocks:
            pair = block(pair, pair_attention_mask=pair_attention_mask)
        if pair.dtype != orig_dtype:
            pair = pair.to(orig_dtype)
        return pair

    folding_trunk.forward = types.MethodType(_no_ckpt_forward, folding_trunk)


def forward_design(model, raw_fwd, features: dict, soft_res_type: torch.Tensor,
                   num_loops: int = 1, num_sampling_steps: int = 2,
                   num_diffusion_samples: int = 1, autocast: bool = True) -> dict:
    """soft res_type 으로 forward. 반환 dict 에 distogram_logits(+pae_logits) grad 포함.
    num_diffusion_samples=1: confidence head 를 1 샘플로(기본 config 32 → 메모리/속도 절약)."""
    other = {k: v for k, v in features.items() if k != "res_type"}
    ctx_amp = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
               if autocast else torch.amp.autocast("cuda", enabled=False))
    with torch.enable_grad(), ctx_amp:
        out = raw_fwd(model, **{**other, "res_type": soft_res_type},
                      num_loops=num_loops, num_sampling_steps=num_sampling_steps,
                      num_diffusion_samples=num_diffusion_samples)
    return out


def enable_pipeline_parallel(model, gpus, handicap: int = 14):
    """트렁크 블록을 여러 GPU 에 분산(pipeline 병렬) — 단일 forward+backward 가 한 GPU 에
    안 들어갈 때(예: STEAP1 trimer L≈1000+, 단일 80GB 천장 L≈850) 사용.

    블록 *경계* 에서만 입력을 그 블록 device 로 .to() 하므로 (연산 내부 cross-device 없음 →
    device_map 이 실패하는 지점을 회피), cross-device backward 를 PyTorch autograd 가 자동
    처리한다(custom autograd/comms 불필요 = research-grade 아님).

      · 각 트렁크 블록(msa_encoder/folding_trunk)을 그리디로 가장 한가한 GPU 에 배치.
        gpus[0](=scaffold: embedding/heads/distogram·loss + 입출력)는 handicap 만큼 블록 적게.
      · msa_encoder 블록은 gradient checkpointing 도 wrap (folding_trunk 는 forward 내부 checkpoint 내장).
      · 각 모듈 출력은 forward_hook 으로 gpus[0] 로 복귀 → 모듈간 인터페이스는 gpus[0] 유지.

    검증(report/_pipeline_probe.py): 2-GPU L1001 peak[75.9,72.2], 3-GPU L1001 peak[56.2,65.5,65.0].
    반환: {device: 배치 블록수}. gpus 예: [0,1,2] (CUDA_VISIBLE_DEVICES 기준 가시 인덱스).
    """
    import torch.utils.checkpoint as _ckpt
    devs = [f"cuda:{g}" if isinstance(g, int) else g for g in gpus]
    dev0 = torch.device(devs[0])

    def _wrap_ckpt(blk):
        orig = blk.forward
        def fwd(*a, **k):
            if torch.is_grad_enabled() and any(torch.is_tensor(x) and x.requires_grad for x in a):
                return _ckpt.checkpoint(orig, *a, use_reentrant=False, **k)
            return orig(*a, **k)
        blk.forward = fwd

    def _wrap_dev(blk, dev):
        blk.to(dev)
        orig = blk.forward
        def fwd(*a, **k):
            a = tuple(x.to(dev) if torch.is_tensor(x) else x for x in a)
            k = {kk: (vv.to(dev) if torch.is_tensor(vv) else vv) for kk, vv in k.items()}
            return orig(*a, **k)
        blk.forward = fwd

    def _out_to0(mod, inp, out):
        def mv(o):
            return o.to(dev0) if (torch.is_tensor(o) and o.device != dev0) else o
        if isinstance(out, tuple):
            return tuple(mv(o) for o in out)
        if isinstance(out, dict):
            return {kk: mv(vv) for kk, vv in out.items()}
        return mv(out)

    lm = getattr(model, "lm_encoder", None)          # lm_encoder: gpus[0] 유지 + checkpoint(작음)
    if lm is not None and getattr(lm, "blocks", None) is not None:
        for blk in lm.blocks:
            _wrap_ckpt(blk)

    loadc = {d: 0 for d in devs}
    loadc[devs[0]] += handicap                       # scaffold 부담 → gpus[0] 핸디캡
    placed = {d: 0 for d in devs}
    for nm in ("msa_encoder", "folding_trunk"):
        mod = getattr(model, nm, None)
        blocks = getattr(mod, "blocks", None) if mod is not None else None
        if blocks is None:
            continue
        self_ckpt = (nm == "folding_trunk")          # folding_trunk 는 forward 내부 checkpoint 내장
        for blk in blocks:
            dev = min(loadc, key=loadc.get)
            loadc[dev] += 1; placed[dev] += 1
            if not self_ckpt:
                _wrap_ckpt(blk)
            _wrap_dev(blk, dev)
        mod.register_forward_hook(_out_to0)
    print(f"[pipeline] 블록 분배 {placed} (handicap={handicap}, gpus={devs})")
    return placed


def enable_recycle_detach_first(model):
    """recycle 의 *마지막 패스만* gradient (BindCraft recycle_mode='last').

    _run_one_loop 은 total_steps(=num_loops+1) 번 z 를 재귀 정제한다. 초기 (total_steps-1)
    패스를 no_grad 로 돌리고 마지막 패스만 그래프에 넣음 →
      · forward 정제는 그대로(2-pass distogram, 논문 loops=1 의 forward 유지)
      · backward 는 1-pass 만 → 속도↑(backward 절반) + 메모리 ~1-pass.
    gradient 는 마지막 패스의 trunk + (z_init/lm_z/msa) 주입 경로로만 흐름(재귀 상태 z 는 detach).
    원본 _run_one_loop 본문을 그대로 재현하되 루프만 no_grad 분기."""
    import types
    import torch.nn.functional as F

    def _run_one_loop_last_grad(self, z, z_init, lm_z, _msa_kwargs, pair_mask, a, b_mat, total_steps):
        lm_cfg = self.config.lm_encoder
        _per = (lm_z is not None and getattr(lm_cfg, "per_loop_lm_dropout", False)
                and getattr(lm_cfg, "lm_dropout", 0.0) > 0.0)
        _p = getattr(lm_cfg, "lm_dropout", 0.0)

        def _iter(z):
            lm_z_i = F.dropout(lm_z, p=_p, training=True) if _per else lm_z
            refined_lm_z = None
            if lm_z_i is not None and self.lm_encoder is not None:
                refined_lm_z = self.lm_encoder(lm_z_i.to(z_init.dtype), pair_attention_mask=pair_mask)
            z_inject_pair = z_init
            if lm_z_i is not None and self.lm_encoder is None:
                z_inject_pair = z_inject_pair + lm_z_i.to(z_inject_pair.dtype)
            if self.msa_encoder is not None and _msa_kwargs is not None:
                msa_pair = self.msa_encoder(x_pair=z_inject_pair, **_msa_kwargs).to(z_inject_pair.dtype)
                z_inject_pair = (msa_pair if self.config.msa_encoder_overwrite
                                 else (z_inject_pair + msa_pair))
            if refined_lm_z is not None:
                z_inject_pair = z_inject_pair + refined_lm_z.to(z_inject_pair.dtype)
            injected_pair = self.parcae_input_norm(z_inject_pair)
            z = a * z + F.linear(injected_pair.to(z.dtype), b_mat)
            z = self.folding_trunk(z, pair_attention_mask=pair_mask)
            return z

        for i in range(total_steps):
            if i < total_steps - 1:               # 초기 패스: no_grad (그래프 X)
                with torch.no_grad():
                    z = _iter(z)
                z = z.detach()
            else:                                 # 마지막 패스: grad
                z = _iter(z)
        return z

    model._run_one_loop = types.MethodType(_run_one_loop_last_grad, model)
    print("[detach-first] recycle 마지막 패스만 grad (BindCraft recycle_mode=last)")
