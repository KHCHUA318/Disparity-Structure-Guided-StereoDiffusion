from pathlib import Path

targets = {
    "ptp_utils.py": [
        "def diffusion_step",
        "model.unet",
        "scheduler.step",
        "noise_pred",
        "latents",
    ],
    "img2stereo.py": [
        "def text2stereoimage_ldm_stable",
        "ptp_utils.diffusion_step",
        "stereo_shift_torch",
        "if i == 10",
        "i > 10",
        "latents_ts",
        "mask",
        "Image.fromarray",
    ],
    "stereoutils.py": [
        "class BNAttention",
        "def forward",
        "direction == 'uni'",
        "direction == 'bi'",
        "attn_batch",
        "def stereo_shift_torch",
        "grid_sample",
        "sacle_factor",
    ],
}

out_lines = []

for file_name, keywords in targets.items():
    path = Path(file_name)
    out_lines.append("\n" + "=" * 100)
    out_lines.append(file_name)
    out_lines.append("=" * 100)

    if not path.exists():
        out_lines.append(f"MISSING: {file_name}")
        continue

    lines = path.read_text(errors="ignore", encoding="utf-8").splitlines()

    for i, line in enumerate(lines, start=1):
        if any(k in line for k in keywords):
            start = max(1, i - 4)
            end = min(len(lines), i + 8)

            out_lines.append(f"\n--- Match around line {i}: {line.strip()} ---")
            for j in range(start, end + 1):
                out_lines.append(f"{j:04d}: {lines[j-1]}")

Path("stereodiffusion_code_inspection.txt").write_text(
    "\n".join(out_lines),
    encoding="utf-8"
)

print("DONE. Output saved to stereodiffusion_code_inspection.txt")
