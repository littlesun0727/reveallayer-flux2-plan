"""Check the Klein dependency stack without downloading model weights.

Default: one logical CUDA device, small fp32/bf16 operations and a tiny random
Flux2 Transformer with LoRA. CPU mode is only a development check, not GPU proof.
"""

import argparse
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import traceback


def check_tensor_ops(torch, device, dtype):
    a = torch.randn(64, 64, device=device, dtype=dtype, requires_grad=True)
    b = torch.randn(64, 64, device=device, dtype=dtype, requires_grad=True)
    conv = torch.nn.Conv2d(3, 4, 3, padding=1).to(device=device, dtype=dtype)
    pixels = torch.randn(1, 3, 16, 16, device=device, dtype=dtype)
    loss = (a @ b).float().square().mean() + conv(pixels).float().square().mean()
    loss.backward()
    gradients = [a.grad, b.grad, conv.weight.grad]
    if not torch.isfinite(loss).item() or any(
        grad is None or not torch.isfinite(grad).all().item()
        or grad.float().abs().sum().item() == 0 for grad in gradients
    ):
        raise RuntimeError("Matrix/convolution backward produced invalid or zero gradients.")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {"passed": True, "dtype": str(dtype), "loss": loss.item()}


def check_klein(torch, device, report):
    import diffusers
    import transformers
    from diffusers import AutoencoderKLFlux2, Flux2KleinPipeline, Flux2Transformer2DModel
    from packaging.requirements import Requirement
    from peft import LoraConfig
    from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

    report["module_paths"] = {
        "diffusers": diffusers.__file__, "transformers": transformers.__file__,
    }
    for cls in (AutoencoderKLFlux2, Flux2KleinPipeline, Flux2Transformer2DModel,
                Qwen2TokenizerFast, Qwen3ForCausalLM):
        if "dummy" in cls.__module__:
            raise RuntimeError("{} is a missing-dependency placeholder.".format(cls.__name__))
    report["klein_imports"] = {"passed": True}

    versions = {}
    mismatches = []
    requirements = Path(__file__).with_name("requirements.txt").read_text(encoding="utf-8")
    for line in requirements.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        requirement = Requirement(line)
        installed = importlib.metadata.version(requirement.name)
        versions[requirement.name] = installed
        if installed not in requirement.specifier:
            mismatches.append("{}: installed {}".format(line, installed))
    report["packages"] = versions
    if mismatches:
        raise RuntimeError("Dependency versions differ from requirements.txt: " + "; ".join(mismatches))

    raw = torch.randn(1, 32, 8, 12, device=device)
    patched = Flux2KleinPipeline._patchify_latents(raw)
    restored = Flux2KleinPipeline._unpatchify_latents(patched)
    if list(patched.shape) != [1, 128, 4, 6] or not torch.equal(raw, restored):
        raise RuntimeError("Klein 32-to-128 channel patchify round trip failed.")
    report["patchify_roundtrip"] = {
        "passed": True, "raw_shape": list(raw.shape), "patched_shape": list(patched.shape),
        "scope": "patchify/unpatchify only; pretrained VAE BN statistics are not tested",
    }

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = Flux2Transformer2DModel(
        in_channels=128, out_channels=128, num_layers=1, num_single_layers=1,
        attention_head_dim=16, num_attention_heads=2, joint_attention_dim=32,
        timestep_guidance_channels=32, axes_dims_rope=(4, 4, 4, 4),
        guidance_embeds=False,
    )
    model.requires_grad_(False)
    model.add_adapter(LoraConfig(r=2, lora_alpha=2, target_modules=["to_q", "to_k", "to_v"]))
    model.to(device=device, dtype=dtype).train()
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any("lora_" not in name for name, _ in trainable):
        raise RuntimeError("Unexpected trainable parameters in the LoRA check.")
    before = [parameter.detach().clone() for _, parameter in trainable]
    optimizer = torch.optim.AdamW([parameter for _, parameter in trainable], lr=1e-3)
    image_tokens = torch.randn(1, 4, 128, device=device, dtype=dtype)
    text_tokens = torch.randn(1, 3, 32, device=device, dtype=dtype)
    image_ids = torch.tensor([[0, 0, 0, 0], [0, 0, 1, 0],
                              [0, 1, 0, 0], [0, 1, 1, 0]], device=device)
    text_ids = torch.zeros(3, 4, device=device)
    text_ids[:, 3] = torch.arange(3, device=device)
    prediction = model(
        hidden_states=image_tokens, encoder_hidden_states=text_tokens,
        timestep=torch.tensor([0.5], device=device, dtype=dtype),
        img_ids=image_ids, txt_ids=text_ids,
    ).sample
    loss = prediction.float().square().mean()
    if prediction.shape != image_tokens.shape or not torch.isfinite(loss).item():
        raise RuntimeError("Tiny Flux2 forward produced invalid output.")
    loss.backward()
    gradients = [parameter.grad for _, parameter in trainable]
    if any(grad is None or not torch.isfinite(grad).all().item() for grad in gradients):
        raise RuntimeError("LoRA gradients are missing or non-finite.")
    gradient_sum = sum(grad.float().abs().sum().item() for grad in gradients)
    if gradient_sum == 0:
        raise RuntimeError("All LoRA gradients are zero.")
    optimizer.step()
    if not any(not torch.equal(old, parameter) for old, (_, parameter) in zip(before, trainable)):
        raise RuntimeError("Optimizer did not update any LoRA parameter.")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    report["tiny_flux2_lora"] = {
        "passed": True, "dtype": str(dtype), "loss": loss.item(),
        "lora_gradient_abs_sum": gradient_sum,
        "trainable_parameters": sum(parameter.numel() for _, parameter in trainable),
        "scope": "random tiny model, not pretrained 4B model or task-quality validation",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0", help="Logical device; e.g. cuda:0 or cpu")
    parser.add_argument("--output", default="checks/klein_environment.json")
    args = parser.parse_args()
    # Set before importing Hugging Face libraries. All model inputs are local/random.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["DIFFUSERS_DISABLE_REMOTE_CODE"] = "1"
    report = {
        "collected_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python_version": sys.version, "device_requested": args.device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "passed": False,
    }
    stage = "torch_import"
    try:
        import torch
        torch.manual_seed(42)
        device = torch.device(args.device)
        if device.type not in ("cpu", "cuda"):
            raise ValueError("Only CPU and CUDA checks are supported.")
        report["torch_version"] = torch.__version__
        report["torch_cuda_build"] = torch.version.cuda
        report["scope"] = "single_gpu" if device.type == "cuda" else "cpu_only"
        stage = "device_setup"
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable in this Python environment.")
            torch.cuda.set_device(device)
            report["gpu_name"] = torch.cuda.get_device_name(device)
            report["visible_gpu_count"] = torch.cuda.device_count()
            torch.cuda.reset_peak_memory_stats(device)
        stage = "fp32_forward_backward"
        report["fp32_ops"] = check_tensor_ops(torch, device, torch.float32)
        if device.type == "cuda":
            stage = "bf16_forward_backward"
            report["bf16_ops"] = check_tensor_ops(torch, device, torch.bfloat16)
        stage = "klein_imports_and_tiny_lora"
        check_klein(torch, device, report)
        if device.type == "cuda":
            report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        report["passed"] = True
    except Exception as error:
        report["failed_stage"] = stage
        report["error"] = str(error)
        report["traceback"] = traceback.format_exc()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("Saved: {}".format(output.resolve()))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
