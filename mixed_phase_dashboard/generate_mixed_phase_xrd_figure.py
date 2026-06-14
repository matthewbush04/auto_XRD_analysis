from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from generate_mixed_phase_dataset import normalize_max, shift_spectrum  # noqa: E402


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/msyhbd.ttc") if bold else Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def choose_battery_sample(data: np.lib.npyio.NpzFile) -> int:
    scenarios = data["battery_scenario"].astype(str)
    indices = np.where(scenarios == "cathode_solid_electrolyte_reference")[0]
    best_idx = int(indices[0])
    best_score = -1.0
    for idx in indices[:400]:
        if int(data["n_phases"][idx]) != 3:
            continue
        fractions = data["component_fractions"][idx]
        minor_sum = float(fractions[1] + fractions[2])
        minor_balance = abs(float(fractions[1] - fractions[2]))
        score = minor_sum - 0.2 * minor_balance
        if score > best_score:
            best_score = score
            best_idx = int(idx)
    return best_idx


def build_component_curves(
    mixed_data: np.lib.npyio.NpzFile,
    single_data: np.lib.npyio.NpzFile,
    sample_idx: int,
) -> np.ndarray:
    n_phases = int(mixed_data["n_phases"][sample_idx])
    source_indices = mixed_data["component_source_indices"][sample_idx]
    fractions = mixed_data["component_fractions"][sample_idx]
    component_shifts = mixed_data["component_shift_bins"][sample_idx]
    global_shift = int(mixed_data["global_shift_bins"][sample_idx])
    scales = mixed_data["component_intensity_scales"][sample_idx]

    curves = []
    for component_idx in range(n_phases):
        curve = normalize_max(single_data["X"][int(source_indices[component_idx])].astype(np.float32))
        curve = shift_spectrum(curve, int(component_shifts[component_idx]))
        curve = float(fractions[component_idx]) * float(scales[component_idx]) * curve
        curve = shift_spectrum(curve, global_shift)
        curves.append(curve)
    curves_array = np.asarray(curves, dtype=np.float32)

    mixture = mixed_data["X"][sample_idx].astype(np.float32)
    summed = np.sum(curves_array, axis=0)
    scale = float(np.max(mixture)) / max(float(np.max(summed)), 1e-8)
    return curves_array * scale


def polyline_points(
    x: np.ndarray,
    y: np.ndarray,
    box: tuple[int, int, int, int],
    y_min: float,
    y_max: float,
) -> list[tuple[int, int]]:
    left, top, right, bottom = box
    x_min = float(x[0])
    x_max = float(x[-1])
    x_coords = left + (x - x_min) / (x_max - x_min) * (right - left)
    y_coords = bottom - (y - y_min) / (y_max - y_min) * (bottom - top)
    x_coords = np.clip(x_coords, left, right).astype(np.int32)
    y_coords = np.clip(y_coords, top, bottom).astype(np.int32)
    return list(zip(x_coords.tolist(), y_coords.tolist()))


def draw_axes(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    x_ticks: list[int],
    y_ticks: list[float],
    font: ImageFont.ImageFont,
) -> None:
    left, top, right, bottom = box
    draw.rectangle(box, outline=(188, 203, 213), width=2)
    for tick in x_ticks:
        x_pos = left + (tick - 10) / 80 * (right - left)
        draw.line((x_pos, top, x_pos, bottom), fill=(235, 240, 243), width=1)
        draw.line((x_pos, bottom, x_pos, bottom + 8), fill=(70, 82, 91), width=2)
        draw.text((x_pos - 18, bottom + 12), str(tick), fill=(70, 82, 91), font=font)
    for tick in y_ticks:
        y_pos = bottom - tick / 1.08 * (bottom - top)
        draw.line((left, y_pos, right, y_pos), fill=(229, 236, 240), width=1)
        draw.line((left - 8, y_pos, left, y_pos), fill=(70, 82, 91), width=2)
        draw.text((left - 58, y_pos - 10), f"{tick:.1f}", fill=(70, 82, 91), font=font)


def draw_rotated_text(
    image: Image.Image,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    bbox = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), text, font=font)
    temp = Image.new("RGBA", (bbox[2] - bbox[0] + 12, bbox[3] - bbox[1] + 12), (255, 255, 255, 0))
    ImageDraw.Draw(temp).text((6, 6), text, font=font, fill=fill)
    temp = temp.rotate(90, expand=True)
    image.alpha_composite(temp, xy)


def main() -> int:
    mixed_path = PROJECT_ROOT / "data" / "processed" / "mixed_phase" / "mixed_v2_hard_eval" / "test_battery_relevant.npz"
    single_path = PROJECT_ROOT / "data" / "processed" / "single_phase" / "test_hard.npz"
    output_dir = PROJECT_ROOT / "figures"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "mixed_phase_xrd_battery_relevant_example.png"

    mixed_data = np.load(mixed_path, allow_pickle=True)
    single_data = np.load(single_path, allow_pickle=True)
    sample_idx = choose_battery_sample(mixed_data)

    two_theta = mixed_data["two_theta"].astype(np.float32)
    mixture = mixed_data["X"][sample_idx].astype(np.float32)
    n_phases = int(mixed_data["n_phases"][sample_idx])
    phase_labels = [str(value) for value in mixed_data["component_phase_labels"][sample_idx][:n_phases]]
    categories = [str(value) for value in mixed_data["component_categories"][sample_idx][:n_phases]]
    fractions = mixed_data["component_fractions"][sample_idx][:n_phases]
    scenario = str(mixed_data["battery_scenario"][sample_idx])
    component_curves = build_component_curves(mixed_data, single_data, sample_idx)

    width, height = 2600, 1500
    image = Image.new("RGBA", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(48, bold=True)
    subtitle_font = load_font(27)
    label_font = load_font(25)
    small_font = load_font(20)
    tiny_font = load_font(17)

    colors = [(23, 105, 170), (31, 138, 91), (183, 121, 31)]
    black = (17, 24, 39)
    muted = (75, 85, 99)
    top_box = (130, 185, 2240, 790)
    bottom_box = (130, 980, 2240, 1285)

    draw.text((130, 58), "电池相关三相混合样品的模拟 XRD 图谱", fill=black, font=title_font)
    draw.text(
        (130, 122),
        "黑线为混相谱线；彩色曲线表示各单相组分按含量缩放后的贡献",
        fill=muted,
        font=subtitle_font,
    )

    draw_axes(draw, top_box, [10, 20, 30, 40, 50, 60, 70, 80, 90], [0.0, 0.5, 1.0], small_font)
    draw.line(polyline_points(two_theta, mixture, top_box, 0.0, 1.08), fill=black, width=4, joint="curve")

    for component_idx in range(n_phases):
        points = polyline_points(two_theta, component_curves[component_idx], top_box, 0.0, 1.08)
        draw.line(points, fill=colors[component_idx] + (190,), width=2)

    draw.text((130, 815), "2θ (degree, Cu Kα)", fill=black, font=label_font)
    draw_rotated_text(image, (32, 360), "归一化强度", label_font, black)

    legend_x = 1510
    legend_y = 218
    legend_w = 700
    legend_h = 178
    draw.rounded_rectangle((legend_x, legend_y, legend_x + legend_w, legend_y + legend_h), radius=12, fill=(255, 255, 255, 238), outline=(200, 212, 220), width=2)
    draw.text((legend_x + 22, legend_y + 18), "组分与含量", fill=black, font=label_font)
    for component_idx in range(n_phases):
        y_pos = legend_y + 58 + component_idx * 34
        draw.line((legend_x + 24, y_pos + 12, legend_x + 74, y_pos + 12), fill=colors[component_idx], width=6)
        text = f"{categories[component_idx]}: {phase_labels[component_idx]}  {fractions[component_idx] * 100:.1f}%"
        draw.text((legend_x + 88, y_pos), text, fill=colors[component_idx], font=small_font)

    draw.text((130, 925), "单相组分谱线（按视觉比例分层显示）", fill=black, font=label_font)
    draw_axes(draw, bottom_box, [10, 30, 50, 70, 90], [], small_font)
    for component_idx in range(n_phases):
        offset = (n_phases - 1 - component_idx) * 0.34
        curve = component_curves[component_idx]
        curve = normalize_max(curve) if float(np.max(curve)) > 0 else curve
        y = curve * 0.26 + offset
        points = polyline_points(two_theta, y, bottom_box, -0.04, n_phases * 0.34)
        draw.line(points, fill=colors[component_idx], width=3, joint="curve")
        y_label = bottom_box[3] - (offset + 0.13 + 0.04) / (n_phases * 0.34 + 0.04) * (bottom_box[3] - bottom_box[1])
        draw.text((2265, int(y_label) - 15), f"{categories[component_idx]}  {fractions[component_idx] * 100:.1f}%", fill=colors[component_idx], font=small_font)

    draw.text((130, 1328), "为什么这张图适合汇报：", fill=black, font=label_font)
    bullets = [
        "展示了混相 XRD 中主相峰与副相峰叠加后的整体 pattern。",
        "彩色组分曲线直观说明模型要从同一条谱线中识别多个 phase。",
        "样本来自 battery-relevant hard evaluation，可对应真实电池材料中的主相 + 电解质/副产物共存场景。",
    ]
    for idx, text in enumerate(bullets):
        draw.text((160, 1370 + idx * 34), f"• {text}", fill=muted, font=small_font)

    footer = f"Data source: mixed_v2_hard_eval/test_battery_relevant.npz, sample #{sample_idx}; scenario = {scenario}"
    draw.text((130, 1460), footer, fill=(107, 114, 128), font=tiny_font)

    image.convert("RGB").save(output_path, quality=95)
    print(f"Saved figure: {output_path}")
    print("Sample:", sample_idx)
    print("Components:")
    for category, label, fraction in zip(categories, phase_labels, fractions):
        print(f"  {category}: {label} ({fraction * 100:.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
