import json
import math
from pathlib import Path


def normalize_privacy_values(layers):
    """Apply per-model linear min-max normalization to layer['privacy'] values.

    Returns True if any value changed, else False.
    """
    privacies = [layer.get("privacy") for layer in layers if isinstance(layer, dict) and isinstance(layer.get("privacy"), (int, float))]
    if not privacies:
        return False

    min_val = min(privacies)
    max_val = max(privacies)
    denom = max_val - min_val

    changed = False
    for layer in layers:
        if not isinstance(layer, dict):
            continue
        val = layer.get("privacy")
        if not isinstance(val, (int, float)):
            continue
        if denom == 0:
            new_val = 0.0
        else:
            new_val = (val - min_val) / denom
        # round to 9 decimals to keep files tidy and stable
        new_val = round(float(new_val), 9)
        if (not math.isclose(val, new_val, rel_tol=0, abs_tol=0)):
            layer["privacy"] = new_val
            changed = True

    return changed


def process_file(path: Path) -> bool:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    layers = data.get("layers")
    if not isinstance(layers, list):
        return False

    changed = normalize_privacy_values(layers)
    if changed:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            f.write("\n")
    return changed


def main():
    root = Path(__file__).resolve().parents[1]
    cfg_dir = root / "configs" / "model_configs"
    targets = sorted(cfg_dir.glob("*.json"))

    any_changed = False
    for p in targets:
        changed = process_file(p)
        print(f"[{'UPDATED' if changed else 'OK'}] {p.relative_to(root)}")
        any_changed = any_changed or changed

    if not any_changed:
        print("No changes needed. All files already normalized.")


if __name__ == "__main__":
    main()


