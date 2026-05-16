"""
=============================================================================
 실제 열화상 PNG 로더 및 배치 FEM 파이프라인
 Real Thermal Image Loader & Batch FEM Pipeline
 길바닥연구소 — SOC종합설계 보조 모듈
=============================================================================
 대상 데이터 (논문 확인):
   출처  : Liu et al., Automation in Construction 161 (2024) 105355
   카메라 : FLUKE TiX580 적외선 카메라 (논문 Section 3)
   해상도 : 640 × 480 px  (논문 Table 1)
   경로  : C:\\Users\\fbsm1\\Downloads\\...\\02-Infrared images
   파일  : 202212195352.png ~ 202212196198.png  (날짜: 2022-12-19)
   이미지 유형: RGB 유사색상 (논문 Fig.4 확인)
   컬러맵 : Rainbow  (FLUKE TiX580 기본 팔레트: 파랑→초록→노랑→빨강)

 ⚠ 온도 범위 주의사항:
   논문에 T_min / T_max 명시 없음.
   FLUKE TiX580 PNG 파일에는 온도 메타데이터가 포함되지 않음.
   → 기본값으로 겨울철 도로 노면 기준 범위 사용 (아래 설정 참고).
   → 정확한 값은 카메라 설정 화면 또는 데이터셋 수집 당시 기록 확인 필요.

 PNG → 온도[°C] 변환 전략:
   ┌──────────────────────────────────────────────────────────────────┐
   │ [모드 A] 그레이스케일 (8-bit / 16-bit)                            │
   │   pixel 0 → T_min,  pixel MAX → T_max  (선형 맵핑)               │
   │                                                                  │
   │ [모드 B] RGB 유사색상 — Rainbow (FLUKE TiX580 기본값 적용)        │
   │   R,G,B → HSV Hue 추출 → rainbow 역변환 → [0,1] → T_min~T_max   │
   │   파랑(H≈240°) = 저온,  빨강(H≈0°) = 고온                        │
   └──────────────────────────────────────────────────────────────────┘
   ※ T_min / T_max 는 카메라 설정 / 현장 기록에서 확인 후 DATASET_T_MIN/MAX 에 입력.
     모를 경우 AUTO 모드로 겨울철 도로 노면 기준 범위 자동 적용.

 파이프라인:
   PNG 로드 → 온도 변환 → 왜곡 보정 (ThermalPreprocessor)
            → FEM 해석 → JSON 결과 → 배치 요약 리포트

 사용 예:
   python thermal_image_loader.py
   (또는)
   from thermal_image_loader import ThermalImageLoader, run_batch_pipeline
   loader = ThermalImageLoader(image_dir=r"C:\\...\\02-Infrared images")
   results = run_batch_pipeline(loader, max_images=20)
=============================================================================
"""

import os
import re
import json
import warnings
import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# ─── 선택적 임포트 (matplotlib) ───────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    warnings.warn("matplotlib 없음 → 시각화 기능 비활성")

# ─── 로컬 모듈 임포트 ─────────────────────────────────────────────────────────
# 같은 폴더에 두 파일을 함께 놓고 실행하세요.
try:
    from thermal_artifact_correction import ThermalPreprocessor
    HAS_CORRECTION = True
except ImportError:
    HAS_CORRECTION = False
    warnings.warn("thermal_artifact_correction.py 없음 → 왜곡 보정 비활성")

try:
    from thermal_stress_fem_system import (
        get_material_properties,
        generate_cst_mesh,
        assemble_and_solve,
        compute_risk_map,
        visualize_results,
        solve_1d_heat_conduction,
        compute_pixel_size_m,
        FLIR_ONE_PRO,
    )
    HAS_FEM = True
except ImportError:
    HAS_FEM = False
    warnings.warn("thermal_stress_fem_system.py 없음 → FEM 해석 비활성")


# =============================================================================
# [설정] FLIR ONE® Pro 카메라 설정 및 온도 범위
# =============================================================================
CAMERA_MODEL     = "FLIR ONE® Pro"
ORIG_WIDTH       = 160              # 열 해상도 (열 픽셀 수)
ORIG_HEIGHT      = 120              # 열 해상도 (행 픽셀 수)
DEFAULT_COLORMAP = "ironbow"        # FLIR ONE Pro 기본 팔레트

# ── 온도 범위 ────────────────────────────────────────────────────────────────
# FLIR ONE Pro 구간 1: -20°C ~ 120°C  (도로 포장 분석 기준)
# FLIR ONE Pro 구간 2:   0°C ~ 400°C  (고온 환경)
#
# [권장] FLIR ONE 앱 또는 FLIR Tools에서 실제 범위 확인 후 직접 입력.
#        Radiometric JPEG → "Export as CSV" 로 절대온도 행렬 직접 추출 가능.
DATASET_T_MIN: Optional[float] = None   # °C ← 확인 후 입력 권장
DATASET_T_MAX: Optional[float] = None   # °C ← 확인 후 입력 권장

# None 시 자동 적용: 도로 포장면 겨울철 기준 (FLIR ONE Pro 구간 1 부분 범위)
_AUTO_T_MIN = -20.0   # °C
_AUTO_T_MAX =  60.0   # °C

# 파일명에서 날짜 파싱 불가 시 기본 촬영일
FALLBACK_CAPTURE_DATE = "2024-01-15"

# FEM 해석 격자 크기 (FLIR ONE Pro 160×120 → 리샘플링)
FEM_ROWS = 40
FEM_COLS = 40

# 기본 촬영 거리 [m]
DEFAULT_DISTANCE_M = 1.0

# 결과 저장 폴더
OUTPUT_DIR = "./thermal_output"


# =============================================================================
# [모듈 1] FLIR ironbow 컬러맵 역변환 LUT
# =============================================================================

def _build_ironbow_lut(n: int = 256) -> np.ndarray:
    """
    FLIR ironbow 컬러맵의 근사 역변환 LUT 생성.
    실제 ironbow 는 비선형이므로 HSV Hue 채널로 근사 역변환한다.

    ironbow 특성:
      value=0   (가장 낮은 온도) → 검정 (H≈240°, S=0 에 가까움)
      value=0.5 (중간)          → 적색 계열 (H≈0~30°)
      value=1.0 (가장 높은 온도) → 흰색

    실용 근사:
      RGB → HSV 변환 후 Hue 채널의 역전된 값으로 intensity 추정
      (파란색 Hue≈240→낮음, 빨간색 Hue≈0→높음)

    Returns
    -------
    lut_rgb : ndarray (n, 3) uint8
        intensity i → [R, G, B]   (0 ≤ i < n)
    """
    import colorsys
    lut = np.zeros((n, 3), dtype=np.uint8)

    for i in range(n):
        t = i / (n - 1)           # 0(저온) → 1(고온)

        # ironbow 근사: 저온=파랑→보라, 중온=빨강, 고온=흰색
        if t < 0.33:
            h = 0.67 - t * 0.67 / 0.33    # blue(0.67) → ~0
            s = 1.0
            v = t / 0.33 * 0.5 + 0.0
        elif t < 0.67:
            h = 0.0
            s = 1.0 - (t - 0.33) / 0.34 * 0.5
            v = 0.5 + (t - 0.33) / 0.34 * 0.35
        else:
            h = 0.0
            s = 0.5 - (t - 0.67) / 0.33 * 0.5
            v = 0.85 + (t - 0.67) / 0.33 * 0.15

        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        lut[i] = [int(r * 255), int(g * 255), int(b * 255)]

    return lut


def _build_rainbow_lut(n: int = 256) -> np.ndarray:
    """
    FLIR rainbow 컬러맵 역변환 LUT.
    blue(저온) → cyan → green → yellow → red(고온)
    HSV: H 240° → 0° 순서
    """
    import colorsys
    lut = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
        t = i / (n - 1)
        h = (1.0 - t) * 0.667      # 240°→0° (blue to red)
        r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
        lut[i] = [int(r * 255), int(g * 255), int(b * 255)]
    return lut


class PseudocolorInverter:
    """
    RGB 유사색상 열화상 이미지를 intensity [0,1] 맵으로 역변환한다.

    원리: 각 픽셀 RGB 에 대해 LUT 상에서 가장 가까운 intensity 값 탐색.
    속도를 위해 H(Hue) 채널만 이용한 근사 역변환을 기본으로 사용하고,
    정밀 모드에서는 전체 LUT 유클리드 거리 탐색을 사용한다.

    Parameters
    ----------
    colormap : str
        'ironbow' (FLIR 기본) 또는 'rainbow'
    fast : bool
        True → Hue 기반 근사 (빠름),  False → LUT 탐색 (정밀)
    """

    def __init__(self, colormap: str = "ironbow", fast: bool = True):
        self.colormap = colormap
        self.fast = fast

        if colormap == "ironbow":
            self._lut = _build_ironbow_lut(256)
        elif colormap == "rainbow":
            self._lut = _build_rainbow_lut(256)
        else:
            raise ValueError(f"지원하지 않는 컬러맵: {colormap} (ironbow / rainbow)")

    def invert(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        img_rgb : ndarray (H, W, 3) uint8

        Returns
        -------
        intensity : ndarray (H, W) float64  [0, 1]
            0 = 가장 낮은 온도, 1 = 가장 높은 온도
        """
        if self.fast:
            return self._fast_hue_invert(img_rgb)
        else:
            return self._lut_search_invert(img_rgb)

    def _fast_hue_invert(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        HSV Hue 채널을 이용한 빠른 역변환.

        ironbow / rainbow 모두 Hue 240°(파랑)=저온, 0°(빨강)=고온 구조.
        Hue ∈ [0, 1]:  intensity ≈ 1 − (hue / 0.667)  [0.667 = 240°/360°]

        단, 고온 영역(흰색, 검정)에서는 채도(S)가 낮아져 Hue 신뢰도↓.
        채도 기반 보정 적용.
        """
        # PIL → float 변환
        img_f = img_rgb.astype(np.float32) / 255.0
        R, G, B = img_f[..., 0], img_f[..., 1], img_f[..., 2]

        V = np.max(img_f, axis=-1)
        m = np.min(img_f, axis=-1)
        S = np.where(V > 1e-6, (V - m) / V, 0.0)

        # Hue 계산
        H = np.zeros_like(V)
        delta = V - m + 1e-9

        mask_r = (V == R) & (S > 0.1)
        mask_g = (V == G) & (S > 0.1)
        mask_b = (V == B) & (S > 0.1)

        H[mask_r] = ((G[mask_r] - B[mask_r]) / delta[mask_r]) % 6.0
        H[mask_g] = (B[mask_g] - R[mask_g]) / delta[mask_g] + 2.0
        H[mask_b] = (R[mask_b] - G[mask_b]) / delta[mask_b] + 4.0
        H = H / 6.0     # → [0, 1]

        # Hue → intensity 역변환
        # 파란색(H≈0.667) = 저온=0,  빨간색(H≈0) = 고온=1
        intensity = np.clip(1.0 - H / 0.667, 0.0, 1.0)

        # 채도 낮은 영역(흰색/검정): Value 로 대체
        low_sat = S < 0.25
        intensity[low_sat] = V[low_sat]   # 밝을수록 고온

        return intensity.astype(np.float64)

    def _lut_search_invert(self, img_rgb: np.ndarray) -> np.ndarray:
        """LUT 유클리드 탐색 (정밀, 느림)."""
        H, W = img_rgb.shape[:2]
        flat = img_rgb.reshape(-1, 3).astype(np.int32)
        lut = self._lut.astype(np.int32)

        intensity = np.zeros(len(flat))
        batch = 1000    # 메모리 절약을 위해 배치 처리
        for start in range(0, len(flat), batch):
            chunk = flat[start:start + batch]          # (B, 3)
            diff  = chunk[:, np.newaxis, :] - lut[np.newaxis, :, :]  # (B, 256, 3)
            dist  = np.sum(diff ** 2, axis=-1)         # (B, 256)
            idx   = np.argmin(dist, axis=-1)           # (B,)
            intensity[start:start + batch] = idx / 255.0

        return intensity.reshape(H, W)


# =============================================================================
# [모듈 2] 온도 범위 자동 추정
# =============================================================================

def estimate_temperature_range(image_dir: str,
                                sample_n: int = 30,
                                mode: str = "grayscale"
                                ) -> tuple[float, float]:
    """
    데이터셋 전체에서 샘플 이미지들을 분석해 온도 범위를 추정한다.

    전략:
    - 그레이스케일: pixel 최솟값/최댓값의 전체 분포에서 1%/99% 백분위수 사용
    - RGB: Hue 역변환 intensity 통계 사용

    Returns
    -------
    (T_min_estimated, T_max_estimated) : tuple[float, float]  [°C]

    Notes
    -----
    추정값은 실제 카메라 설정과 다를 수 있으므로,
    논문 / 카메라 메타데이터로 확인 후 DATASET_T_MIN/T_MAX 에 직접 입력 권장.
    """
    files = sorted(Path(image_dir).glob("*.png"))
    if not files:
        raise FileNotFoundError(f"PNG 파일 없음: {image_dir}")

    step = max(1, len(files) // sample_n)
    sampled = files[::step][:sample_n]

    all_vals = []
    for f in sampled:
        try:
            img = Image.open(f)
            arr = np.array(img)
            if arr.ndim == 2:          # 그레이스케일
                all_vals.append(arr.flatten().astype(np.float64))
            elif arr.shape[2] >= 3:    # RGB
                # Hue 역변환으로 상대 intensity 만 수집
                inv = PseudocolorInverter("ironbow", fast=True)
                intens = inv.invert(arr[:, :, :3])
                all_vals.append(intens.flatten())
        except Exception:
            continue

    if not all_vals:
        return _AUTO_T_MIN, _AUTO_T_MAX

    combined = np.concatenate(all_vals)
    pct_lo = float(np.percentile(combined, 1))
    pct_hi = float(np.percentile(combined, 99))

    if mode in ("grayscale",):
        # 그레이스케일 픽셀값 단위 → 온도 변환 불가
        # FLUKE TiX580 겨울철 도로 노면 기준 적용
        T_min_est = _AUTO_T_MIN
        T_max_est = _AUTO_T_MAX
    else:
        # RGB rainbow: intensity [0,1] → 온도 범위도 동일 기준 적용
        T_min_est = _AUTO_T_MIN
        T_max_est = _AUTO_T_MAX

    print(f"[온도 범위 AUTO 추정] 샘플 {len(sampled)}장 분석 완료")
    print(f"  픽셀 통계: [{pct_lo:.3f}, {pct_hi:.3f}]  (intensity 단위, 0~1)")
    print(f"  적용 온도 범위: {T_min_est}°C ~ {T_max_est}°C")
    print(f"  카메라: {CAMERA_MODEL} / 컬러맵: {DEFAULT_COLORMAP}")
    print("!" * 65)
    print("  [FEM 신뢰도 경고] AUTO 온도 범위 사용 중")
    print("  PNG 유사색상 → 온도 역변환은 카메라 오토-스케일 구간에")
    print("  의존합니다. T_min/T_max 가 실제와 다르면 FEM 열하중이")
    print("  수십 배 왜곡되어 해석 결과를 신뢰할 수 없습니다.")
    print("  ─ 권장 해결책 ──────────────────────────────────────")
    print("  FLIR Tools → File → Export → CSV 로 절대온도 직접 추출")
    print("  후 csv_to_temperature() 함수를 사용하십시오.")
    print("!" * 65)

    return T_min_est, T_max_est


# =============================================================================
# [모듈 3] 단일 PNG → 온도 행렬 변환
# =============================================================================

def png_to_temperature(
    png_path: str,
    T_min: float,
    T_max: float,
    target_rows: int = FEM_ROWS,
    target_cols: int = FEM_COLS,
    colormap: str = "auto",
) -> tuple[np.ndarray, dict]:
    """
    단일 PNG 파일을 절대온도 행렬 [°C] 로 변환한다.

    ⚠ FEM 신뢰도 주의
    ------------------
    이 함수는 유사색상 역변환에 의존합니다.
    T_min / T_max 가 카메라 촬영 당시의 실제 스케일과 다를 경우,
    온도차가 수십 배 왜곡되어 FEM 열하중 벡터가 완전히 틀어집니다.
    가능하면 csv_to_temperature() 를 1순위로 사용하십시오.

    Parameters
    ----------
    png_path   : str         PNG 파일 전체 경로
    T_min      : float [°C]  픽셀 0 (intensity 0) 에 대응하는 온도
    T_max      : float [°C]  픽셀 MAX (intensity 1) 에 대응하는 온도
    target_rows: int         FEM 격자 행 수 (리샘플링 대상)
    target_cols: int         FEM 격자 열 수
    colormap   : str         'auto' | 'grayscale' | 'ironbow' | 'rainbow'

    Returns
    -------
    T    : ndarray (target_rows, target_cols) [°C]
    meta : dict  {filename, original_size, bit_depth, mode_detected, T_min, T_max}
    """
    # T_min/T_max 가 AUTO 기본값이면 FEM 신뢰도 경고
    if T_min is None or T_max is None or T_min == _AUTO_T_MIN or T_max == _AUTO_T_MAX:
        warnings.warn(
            f"[png_to_temperature] '{os.path.basename(png_path)}': "
            "T_min/T_max 가 AUTO 기본값입니다. "
            "FEM 열하중 신뢰도를 보장하려면 실측 온도 범위를 직접 입력하거나 "
            "CSV 파일을 사용하십시오.",
            stacklevel=2
        )
    T_min = T_min if T_min is not None else _AUTO_T_MIN
    T_max = T_max if T_max is not None else _AUTO_T_MAX
    img   = Image.open(png_path)
    fname = os.path.basename(png_path)
    meta  = {
        "filename"     : fname,
        "original_size": img.size,   # (W, H) = (160, 120) for FLIR ONE Pro
        "T_min"        : T_min,
        "T_max"        : T_max,
        "camera"       : CAMERA_MODEL,
    }

    # FLIR ONE Pro 해상도 검증
    w, h = img.size
    if (w, h) != (ORIG_WIDTH, ORIG_HEIGHT):
        warnings.warn(
            f"[png_to_temperature] 원본 크기 {w}×{h}가 "
            f"FLIR ONE Pro 열 해상도 {ORIG_WIDTH}×{ORIG_HEIGHT}와 다릅니다.\n"
            f"  VividIR™ 업스케일 이미지이거나 다른 카메라 이미지일 수 있습니다."
        )

    arr = np.array(img)

    # ── 이미지 모드 판별 ──────────────────────────────────────────────────────
    if colormap == "auto":
        if arr.ndim == 2:
            detected = "grayscale"
        elif arr.shape[2] == 4:
            arr = arr[:, :, :3]
            if np.mean(np.abs(arr[:, :, 0].astype(int) -
                              arr[:, :, 1].astype(int))) < 3:
                detected = "grayscale"
                arr = arr[:, :, 0]
            else:
                detected = "ironbow"   # FLIR ONE Pro 기본값
        else:
            if np.mean(np.abs(arr[:, :, 0].astype(int) -
                              arr[:, :, 1].astype(int))) < 3:
                detected = "grayscale"
                arr = arr[:, :, 0]
            else:
                detected = "ironbow"   # FLIR ONE Pro 기본값 (원본 rgb_color → ironbow)
    else:
        detected = colormap
        if colormap == "grayscale" and arr.ndim == 3:
            arr = arr[:, :, 0]

    meta["mode_detected"] = detected
    meta["bit_depth"]     = arr.dtype.itemsize * 8

    # ── pixel → intensity [0, 1] ──────────────────────────────────────────────
    if detected == "grayscale":
        arr_f   = arr.astype(np.float64)
        max_val = float(np.iinfo(arr.dtype).max) if np.issubdtype(arr.dtype, np.integer) else 1.0
        intensity = arr_f / max_val

    elif detected in ("ironbow", "rainbow"):
        inverter = PseudocolorInverter(detected, fast=True)
        arr_rgb  = arr[:, :, :3] if arr.ndim == 3 else np.stack([arr]*3, axis=-1)
        intensity = inverter.invert(arr_rgb.astype(np.uint8))

    else:
        raise ValueError(f"알 수 없는 colormap 모드: {detected}")

    # ── intensity → 온도 ──────────────────────────────────────────────────────
    T_full = T_min + intensity * (T_max - T_min)

    # FLIR ONE Pro 유효 온도 범위 클리핑 (-20°C ~ 400°C)
    T_full = np.clip(T_full, -20.0, 400.0)

    # ── FEM 격자 크기로 리샘플링 (PIL.LANCZOS) ────────────────────────────────
    T_pil     = Image.fromarray(T_full.astype(np.float32))
    T_resized = T_pil.resize((target_cols, target_rows), Image.LANCZOS)
    T         = np.array(T_resized, dtype=np.float64)

    meta["T_range_min"] = round(float(T.min()), 2)
    meta["T_range_max"] = round(float(T.max()), 2)
    return T, meta


# =============================================================================
# [모듈 3.5] CSV → 절대온도 행렬 (1순위, FEM 신뢰도 HIGH)
# =============================================================================

def csv_to_temperature(
    csv_path   : str,
    target_rows: int = FEM_ROWS,
    target_cols: int = FEM_COLS,
) -> tuple[np.ndarray, dict]:
    """
    FLIR Tools / FLIR Ignite Export → CSV 의 절대온도 행렬을 로드한다.

    CSV 포맷 가정:
      - 행: 열화상 픽셀 행 (top→bottom)
      - 열: 열화상 픽셀 열 (left→right)
      - 값: 절대 온도 [°C] (방사성 측정값)
      - 헤더 / 메타 행은 자동 감지·스킵

    PNG 폴백과의 차이
    -----------------
    이 경로로 로드된 데이터는 카메라가 픽셀 단위로 출력한 절대 온도 그
    자체이므로, 임의 스케일링 위험이 없다. FEM 열하중 벡터의 신뢰도가
    가장 높은 입력원이다 (load_mode='csv_absolute').

    Parameters
    ----------
    csv_path   : str  CSV 파일 전체 경로
    target_rows: int  FEM 격자 행 수 (리샘플링 대상)
    target_cols: int  FEM 격자 열 수

    Returns
    -------
    T    : ndarray (target_rows, target_cols) [°C]
    meta : dict  {filename, original_size, T_range_min/max, source}
    """
    fname = os.path.basename(csv_path)
    raw = None
    last_err = None
    # 헤더 행 0~5 시도 (FLIR Tools 는 보통 0 또는 9~10 행 헤더)
    for skip in range(0, 11):
        try:
            arr = np.genfromtxt(
                csv_path, delimiter=",", skip_header=skip,
                dtype=np.float64, invalid_raise=False, encoding="utf-8"
            )
            if arr.ndim == 2 and arr.shape[0] >= 4 and arr.shape[1] >= 4:
                # NaN 비율이 너무 높으면 헤더 잘못 잡힌 것
                nan_ratio = float(np.isnan(arr).mean())
                if nan_ratio < 0.5:
                    raw = arr
                    break
        except Exception as e:
            last_err = e
            continue

    if raw is None:
        raise ValueError(
            f"[csv_to_temperature] CSV 파싱 실패: {fname}\n"
            f"  마지막 오류: {last_err}\n"
            "  CSV 포맷이 (rows × cols) 절대온도 행렬인지 확인하세요."
        )

    # NaN 픽셀은 행 평균으로 대체 (소량 누락 보완)
    if np.isnan(raw).any():
        col_mean = np.nanmean(raw, axis=0, keepdims=True)
        nan_mask = np.isnan(raw)
        raw[nan_mask] = np.broadcast_to(col_mean, raw.shape)[nan_mask]
        # 그래도 NaN 이 남으면 전역 중앙값 대체
        if np.isnan(raw).any():
            raw[np.isnan(raw)] = float(np.nanmedian(raw))

    # FLIR ONE Pro 유효 온도 범위 검증
    if raw.min() < -50.0 or raw.max() > 500.0:
        warnings.warn(
            f"[csv_to_temperature] {fname}: 온도 범위 이상 "
            f"({raw.min():.1f} ~ {raw.max():.1f}°C). 단위가 °C 인지 확인하세요."
        )

    H, W = raw.shape

    # FEM 격자 크기로 리샘플링 (PIL.LANCZOS, float32 한정)
    T_pil     = Image.fromarray(raw.astype(np.float32))
    T_resized = T_pil.resize((target_cols, target_rows), Image.LANCZOS)
    T         = np.array(T_resized, dtype=np.float64)

    meta = {
        "filename"     : fname,
        "source"       : "csv_absolute",
        "original_size": (W, H),
        "camera"       : CAMERA_MODEL,
        "T_range_min"  : round(float(T.min()), 2),
        "T_range_max"  : round(float(T.max()), 2),
        "T_raw_min"    : round(float(raw.min()), 2),
        "T_raw_max"    : round(float(raw.max()), 2),
    }
    return T, meta


# =============================================================================
# [모듈 4] 파일명 파서 (날짜 / 시퀀스 추출)
# =============================================================================

def parse_filename(fname: str) -> dict:
    """
    파일명 패턴 분석.

    지원 패턴:
      202401150001.png  → date='2024-01-15', seq=1
      20240115_001.png  → date='2024-01-15', seq=1
      FLIR0001.png      → seq=1  (FLIR ONE Pro 앱 기본 파일명)
      (그 외 패턴)      → capture_date=FALLBACK_CAPTURE_DATE

    Returns
    -------
    info : dict {stem, date_str, seq, capture_date}
    """
    stem = Path(fname).stem

    # 패턴 1: 8자리 날짜 + 임의 숫자 (202401150001)
    m = re.match(r"^(\d{4})(\d{2})(\d{2})(\d+)$", stem)
    if m:
        year, month, day, seq_str = m.groups()
        try:
            dt = datetime.date(int(year), int(month), int(day))
            return {
                "stem"        : stem,
                "date_str"    : dt.isoformat(),
                "seq"         : int(seq_str),
                "capture_date": dt.isoformat(),
            }
        except ValueError:
            pass

    # 패턴 2: FLIR + 숫자 (FLIR ONE Pro 앱 기본 파일명: FLIR0001, FLIR0002 ...)
    m2 = re.match(r"^(?:FLIR|flir)(\d+)$", stem)
    if m2:
        return {
            "stem"        : stem,
            "date_str"    : None,
            "seq"         : int(m2.group(1)),
            "capture_date": FALLBACK_CAPTURE_DATE,
        }

    # 패턴 3: 날짜_시퀀스 (20240115_001)
    m3 = re.match(r"^(\d{4})(\d{2})(\d{2})_(\d+)$", stem)
    if m3:
        year, month, day, seq_str = m3.groups()
        try:
            dt = datetime.date(int(year), int(month), int(day))
            return {
                "stem"        : stem,
                "date_str"    : dt.isoformat(),
                "seq"         : int(seq_str),
                "capture_date": dt.isoformat(),
            }
        except ValueError:
            pass

    # 패턴 불일치 → 기본값
    return {
        "stem"        : stem,
        "date_str"    : None,
        "seq"         : None,
        "capture_date": FALLBACK_CAPTURE_DATE,
    }


# =============================================================================
# [모듈 5] ThermalImageLoader (파일 목록 관리 + 개별 로드)
# =============================================================================

class ThermalImageLoader:
    """
    열화상 PNG 디렉토리를 관리하고 개별 이미지를 온도 행렬로 변환하는 클래스.

    Parameters
    ----------
    image_dir    : str         PNG 파일이 있는 디렉토리 경로
    T_min        : float|None  픽셀 최솟값에 대응하는 온도 [°C]. None=AUTO
    T_max        : float|None  픽셀 최댓값에 대응하는 온도 [°C]. None=AUTO
    colormap     : str         'auto'|'grayscale'|'ironbow'|'rainbow'
    target_rows  : int         FEM 격자 행 수
    target_cols  : int         FEM 격자 열 수
    seq_range    : tuple|None  (seq_min, seq_max) 로드할 시퀀스 번호 범위.
                               None 이면 전체 로드.

    주요 속성
    ---------
    files        : list[Path]  정렬된 PNG 경로 목록
    T_min, T_max : float       적용 중인 온도 범위
    """

    def __init__(self,
                 image_dir   : str,
                 T_min       : Optional[float] = DATASET_T_MIN,
                 T_max       : Optional[float] = DATASET_T_MAX,
                 colormap    : str             = DEFAULT_COLORMAP,   # "ironbow" (FLIR ONE Pro)
                 target_rows : int             = FEM_ROWS,
                 target_cols : int             = FEM_COLS,
                 seq_range   : Optional[tuple] = None,
                 distance_m  : float           = DEFAULT_DISTANCE_M):

        self.image_dir   = image_dir
        self.colormap    = colormap
        self.target_rows = target_rows
        self.target_cols = target_cols
        self.seq_range   = seq_range
        self.distance_m  = distance_m

        self.files = self._collect_files()
        print(f"[Loader] 카메라  : {CAMERA_MODEL}")
        print(f"[Loader] 디렉토리: {image_dir}")
        ext_label = "PNG" if (self.files and self.files[0].suffix.lower() == ".png") else "CSV"
        print(f"[Loader] 탐지 파일: {len(self.files)}장 ({ext_label})")
        print(f"[Loader] 팔레트  : {colormap}  |  FEM 격자: {target_cols}×{target_rows}")
        print(f"[Loader] 촬영거리: {distance_m:.1f}m")

        # CSV-only 모드 자동 감지 (PNG 0장이고 CSV가 있을 때)
        is_csv_only = bool(self.files) and all(
            f.suffix.lower() == ".csv" for f in self.files
        )
        self.is_csv_only = is_csv_only

        if not self.files:
            raise FileNotFoundError(
                f"열화상 파일 없음 (PNG/CSV 둘 다): {image_dir}"
            )

        if is_csv_only:
            # CSV는 절대온도라서 T_min/T_max 추정 불필요
            self.T_min = None
            self.T_max = None
            print(f"[Loader] CSV-only 모드 — 온도는 CSV 절대값 사용")
        elif T_min is not None and T_max is not None:
            self.T_min, self.T_max = T_min, T_max
            print(f"[Loader] 온도 범위 (수동): {T_min}°C ~ {T_max}°C")
        else:
            print("[Loader] 온도 범위 AUTO 추정 중...")
            self.T_min, self.T_max = estimate_temperature_range(
                image_dir, sample_n=min(20, len(self.files)), mode=colormap
            )

    def _collect_files(self) -> list:
        """
        seq_range 필터를 적용해 입력 파일 목록 수집·정렬.

        파일 우선순위
        -------------
        1) PNG가 있으면 PNG 목록 반환 (load_one()이 같은 stem의 .csv를 우선 사용)
        2) PNG가 없으면 CSV 목록을 직접 반환 (CSV-only 데이터셋 지원)
        """
        all_pngs = sorted(Path(self.image_dir).glob("*.png"))
        if all_pngs:
            all_files = all_pngs
        else:
            # PNG가 전혀 없으면 CSV-only 데이터셋으로 간주
            all_files = sorted(Path(self.image_dir).glob("*.csv"))

        if self.seq_range is None:
            return all_files

        seq_min, seq_max = self.seq_range
        filtered = []
        for f in all_files:
            info = parse_filename(f.name)
            if info["seq"] is not None:
                if seq_min <= info["seq"] <= seq_max:
                    filtered.append(f)
            else:
                filtered.append(f)   # 패턴 불일치는 포함
        return filtered

    def __len__(self):
        return len(self.files)

    def load_one(self, index: int) -> tuple[np.ndarray, dict]:
        """
        index 번째 파일을 로드해 온도 행렬과 메타데이터 반환.

        로드 우선순위
        -------------
        1순위 (권장) — CSV 절대온도 파일
            같은 stem의 .csv 가 존재하면 csv_to_temperature() 호출.
            온도 범위 입력 불필요, FEM 신뢰도 최고.

        2순위 (폴백) — PNG 유사색상 역변환
            CSV 없을 때만 사용. T_min/T_max 가 None 이거나
            AUTO 기본값인 경우 강력 경고를 출력하고 진행.

        ⚠ 경고
        -------
        PNG 유사색상 역변환은 카메라 화면의 오토-스케일 구간을 알 수
        없을 때 실제 온도차를 수십 배 부풀릴 수 있습니다.
        FEM 신뢰도를 보장하려면 반드시 CSV를 사용하십시오.
        """
        f = self.files[index]

        # ── CSV 파일이 직접 입력된 경우 (CSV-only 데이터셋) ────────────────
        if f.suffix.lower() == ".csv":
            T, meta = csv_to_temperature(
                str(f), self.target_rows, self.target_cols
            )
            meta["load_mode"] = "csv_absolute"   # FEM 신뢰도: HIGH
            file_info = parse_filename(f.name)
            meta.update(file_info)
            meta["distance_m"] = self.distance_m
            return T, meta

        csv_path = f.with_suffix(".csv")

        # ── 1순위: 동명 CSV 파일 ───────────────────────────────────────────
        if csv_path.exists():
            T, meta = csv_to_temperature(
                str(csv_path), self.target_rows, self.target_cols
            )
            meta["load_mode"] = "csv_absolute"   # FEM 신뢰도: HIGH

        # ── 2순위: PNG 유사색상 (폴백) ───────────────────────────────────
        else:
            # T_min/T_max 가 AUTO 기본값이거나 미입력인 경우 강력 경고
            t_min_is_auto = (self.T_min is None or self.T_min == _AUTO_T_MIN)
            t_max_is_auto = (self.T_max is None or self.T_max == _AUTO_T_MAX)

            if t_min_is_auto or t_max_is_auto:
                warnings.warn(
                    "\n" + "!" * 65 + "\n"
                    f"  [FEM 신뢰도 경고] {f.name}\n"
                    "  CSV 파일 없음 → PNG 유사색상 역변환(폴백) 사용 중.\n"
                    "  T_min / T_max 가 AUTO 기본값으로 설정되어 있습니다.\n"
                    "  카메라 오토-스케일 구간이 불명확하면 실제 온도차가\n"
                    "  수십 배 과장되어 FEM 하중 벡터가 완전히 왜곡됩니다.\n"
                    "  ─ 해결 방법 (둘 중 하나) ─────────────────────────\n"
                    "  A) FLIR Tools → Export → CSV 로 절대온도 파일 추출\n"
                    "  B) ThermalImageLoader(T_min=실측값, T_max=실측값) 입력\n"
                    "!" * 65,
                    stacklevel=3
                )

            T, meta = png_to_temperature(
                str(f), self.T_min, self.T_max,
                self.target_rows, self.target_cols, self.colormap
            )
            meta["load_mode"] = "png_pseudocolor_fallback"   # FEM 신뢰도: LOW

        file_info = parse_filename(f.name)
        meta.update(file_info)
        meta["distance_m"] = self.distance_m
        return T, meta

    def load_batch(self, indices=None):
        """
        여러 파일을 순서대로 로드.
        indices=None 이면 전체 로드.

        Yields
        ------
        (idx, T, meta)
        """
        if indices is None:
            indices = range(len(self.files))
        for i in indices:
            try:
                T, meta = self.load_one(i)
                yield i, T, meta
            except Exception as e:
                warnings.warn(f"[Loader] {self.files[i].name} 로드 실패: {e}")


# =============================================================================
# [모듈 6] 단일 이미지 FEM 파이프라인
# =============================================================================

def run_single_fem(T_raw: np.ndarray,
                   capture_date    : str,
                   apply_correction: bool  = True,
                   output_dir      : str   = OUTPUT_DIR,
                   save_fig        : bool  = True,
                   stem            : str   = "image",
                   distance_m      : float = DEFAULT_DISTANCE_M,
                   bc_mode         : str   = "symmetric",
                   winkler_k       : float = 0.0) -> dict:
    """
    단일 온도 행렬에 대해 전처리 → FEM 해석 → 결과 반환.

    Parameters
    ----------
    T_raw           : ndarray (rows, cols) [°C]  로드된 원본 온도 행렬
    capture_date    : str 'YYYY-MM-DD'            촬영일
    apply_correction: bool                         왜곡 보정 적용 여부
    output_dir      : str                          결과 저장 폴더
    save_fig        : bool                         FEM 결과 이미지 저장 여부
    stem            : str                          출력 파일 접두사
    distance_m      : float                        촬영 거리 [m] (픽셀 크기 계산용)

    Returns
    -------
    result : dict  FEM 해석 결과 (JSON 호환)
    """
    if not HAS_FEM:
        raise RuntimeError("thermal_stress_fem_system.py 를 같은 폴더에 배치하세요.")

    rows, cols = T_raw.shape
    os.makedirs(output_dir, exist_ok=True)

    # [1] 물성치
    props = get_material_properties(capture_date)

    # [2] 왜곡 보정
    if apply_correction and HAS_CORRECTION:
        prep           = ThermalPreprocessor(rows, cols)
        T_clean        = prep.correct(T_raw)
        correction_log = prep.correction_summary()
    else:
        T_clean        = T_raw.copy()
        correction_log = "보정 미적용"

    # [3] FEM 메시 생성 및 풀이 (HFOV/VFOV 기반 픽셀 물리 크기 자동 적용)
    nodes, elements = generate_cst_mesh(rows, cols, distance_m=distance_m)
    T_field = T_clean.flatten()
    d_global, elem_stresses, elem_principal = assemble_and_solve(
        nodes, elements, T_field, props,
        bc_mode=bc_mode, winkler_k=winkler_k,
    )

    # [4] 위험도
    risk_map   = compute_risk_map(elem_principal, props["S_t"])
    n_critical = int(np.sum(risk_map >= 1.0))
    n_warning  = int(np.sum((risk_map >= 0.7) & (risk_map < 1.0)))
    n_total    = len(elements)

    if n_critical > 0:
        grade = "Critical"
    elif n_warning > 0:
        grade = "Warning"
    else:
        grade = "Safe"

    # [5] 시각화 — thermal_stress_fem_system.visualize_results 재사용
    #     (버그 수정: 이전 _save_fem_figure 중복 제거, fig_path 미정의 버그 수정)
    fig_path = ""
    if save_fig and HAS_FEM and HAS_MPL:
        fig_path = visualize_results(
            nodes, elements, T_field, elem_principal, risk_map,
            rows, cols, props,
            output_dir=output_dir,
            distance_m=distance_m,
        )

    result = {
        "stem"         : stem,
        "camera"       : CAMERA_MODEL,
        "capture_date" : capture_date,
        "distance_m"   : distance_m,
        "T_raw_range"  : [round(float(T_raw.min()), 2), round(float(T_raw.max()), 2)],
        "T_clean_range": [round(float(T_clean.min()), 2), round(float(T_clean.max()), 2)],
        "correction"   : correction_log,
        "material"     : {
            "E_MPa"  : props["E"],
            "nu"     : props["nu"],
            "alpha"  : props["alpha"],
            "S_t_MPa": props["S_t"],
        },
        "fem"    : {
            "max_stress_MPa": round(float(np.max(np.abs(elem_principal))), 4),
            "max_ratio"     : round(float(risk_map.max()), 4),
            "critical"      : n_critical,
            "warning"       : n_warning,
            "total"         : n_total,
        },
        "grade"   : grade,
        "fig_path": fig_path,
    }
    return result



# =============================================================================
# [모듈 7] 배치 파이프라인 + 요약 리포트
# =============================================================================

def run_batch_pipeline(loader: ThermalImageLoader,
                       max_images: Optional[int] = None,
                       apply_correction: bool = True,
                       save_figs: bool = True,
                       output_dir: str = OUTPUT_DIR) -> list[dict]:
    """
    디렉토리 내 전체 (또는 max_images 장) 이미지에 대해 FEM 파이프라인 실행.

    Parameters
    ----------
    loader         : ThermalImageLoader
    max_images     : int|None   처리할 최대 이미지 수. None=전체
    apply_correction: bool      왜곡 보정 적용 여부
    save_figs      : bool       FEM 결과 이미지 저장 여부
    output_dir     : str        결과 저장 폴더

    Returns
    -------
    results : list[dict]   이미지별 FEM 결과 목록
    """
    os.makedirs(output_dir, exist_ok=True)
    n = len(loader) if max_images is None else min(max_images, len(loader))
    results = []

    print(f"\n{'='*60}")
    print(f"  배치 FEM 파이프라인 시작  [{CAMERA_MODEL}]  ({n}장 처리 예정)")
    print(f"  출력 폴더: {output_dir}")
    print(f"{'='*60}\n")

    for idx, T_raw, meta in loader.load_batch(range(n)):
        stem          = meta.get("stem", f"img_{idx:04d}")
        capture_date  = meta.get("capture_date", FALLBACK_CAPTURE_DATE)
        mode_detected = meta.get("mode_detected", "?")
        original_size = meta.get("original_size", "?")
        distance_m    = meta.get("distance_m", DEFAULT_DISTANCE_M)

        print(f"[{idx+1:4d}/{n}] {stem}  "
              f"원본:{original_size}  모드:{mode_detected}  날짜:{capture_date}  "
              f"거리:{distance_m:.1f}m")

        if not HAS_FEM:
            print("  ⚠ FEM 모듈 없음 → 건너뜀")
            continue

        try:
            res = run_single_fem(
                T_raw, capture_date,
                apply_correction=apply_correction,
                output_dir=output_dir,
                save_fig=save_figs,
                stem=stem,
                distance_m=distance_m,
            )
            res["index"] = idx
            results.append(res)

            grade_icon = {"Critical": "🔴", "Warning": "🟡", "Safe": "🟢"}.get(res["grade"], "⚪")
            print(f"        → {grade_icon} {res['grade']:8s}  "
                  f"σ_max={res['fem']['max_stress_MPa']:.3f} MPa  "
                  f"R_max={res['fem']['max_ratio']:.3f}  "
                  f"Critical요소={res['fem']['critical']}")

        except Exception as e:
            print(f"        → ❌ 오류: {e}")

    # ── 배치 요약 리포트 생성 ─────────────────────────────────────────────────
    if results:
        _save_batch_report(results, output_dir)

    print(f"\n✅ 배치 처리 완료: {len(results)}/{n}장 성공")
    return results


def _save_batch_report(results: list[dict], output_dir: str):
    """배치 결과를 JSON + 요약 텍스트 + 시각화로 저장."""

    # JSON
    json_path = os.path.join(output_dir, "batch_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 텍스트 요약
    n = len(results)
    grades = [r["grade"] for r in results]
    n_crit = grades.count("Critical")
    n_warn = grades.count("Warning")
    n_safe = grades.count("Safe")

    max_ratios = [r["fem"]["max_ratio"] for r in results]
    top5 = sorted(results, key=lambda x: x["fem"]["max_ratio"], reverse=True)[:5]

    lines = [
        "=" * 65,
        "  배치 FEM 해석 요약 보고서",
        "=" * 65,
        f"  총 처리 이미지 : {n}장",
        f"  🔴 파괴위험    : {n_crit}장 ({n_crit/n*100:.1f}%)",
        f"  🟡 경고        : {n_warn}장 ({n_warn/n*100:.1f}%)",
        f"  🟢 안전        : {n_safe}장 ({n_safe/n*100:.1f}%)",
        "",
        f"  최대 응력비 평균 : {np.mean(max_ratios):.4f}",
        f"  최대 응력비 최대 : {np.max(max_ratios):.4f}",
        f"  최대 응력비 최소 : {np.min(max_ratios):.4f}",
        "",
        "  ─ 위험도 상위 5장 ─",
    ]
    for r in top5:
        lines.append(
            f"    {r['stem']:20s}  R_max={r['fem']['max_ratio']:.4f}  "
            f"σ={r['fem']['max_stress_MPa']:.3f} MPa  [{r['grade']}]"
        )
    lines.append("=" * 65)
    report_txt = "\n".join(lines)

    txt_path = os.path.join(output_dir, "batch_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_txt)
    print(f"\n{report_txt}")

    # 시각화: 응력비 분포 히스토그램
    if HAS_MPL and n >= 3:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # (a) 응력비 히스토그램
        ax = axes[0]
        ax.hist(max_ratios, bins=min(30, n), color="steelblue", edgecolor="white", alpha=0.85)
        ax.axvline(0.7, color="orange", lw=1.5, linestyle="--", label="Warning (0.7)")
        ax.axvline(1.0, color="red",    lw=1.5, linestyle="--", label="Critical (1.0)")
        ax.set_xlabel("Max Stress Ratio")
        ax.set_ylabel("Image Count")
        ax.set_title("(a) Distribution of Max Stress Ratio")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # (b) 등급 파이차트
        ax = axes[1]
        sizes  = [n_crit, n_warn, n_safe]
        labels = [f"Critical\n({n_crit})", f"Warning\n({n_warn})", f"Safe\n({n_safe})"]
        colors = ["#e31a1c", "#ffa500", "#33a02c"]
        non_zero = [(s, l, c) for s, l, c in zip(sizes, labels, colors) if s > 0]
        if non_zero:
            sz, lb, cl = zip(*non_zero)
            ax.pie(sz, labels=lb, colors=cl, autopct="%1.1f%%", startangle=90)
        ax.set_title("(b) Risk Grade Distribution")

        fig.suptitle(f"Batch FEM Analysis Summary  ({n} images)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        hist_path = os.path.join(output_dir, "batch_summary_plot.png")
        plt.savefig(hist_path, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"[리포트] 히스토그램 저장 → {hist_path}")

    print(f"[리포트] JSON → {json_path}")
    print(f"[리포트] TXT  → {txt_path}")


# =============================================================================
# [단독 실행] 데모 / 실제 데이터 실행 진입점
# =============================================================================

if __name__ == "__main__":

    # ──────────────────────────────────────────────────────────────────────────
    # ▼▼▼ 여기만 수정하세요 ▼▼▼
    # ──────────────────────────────────────────────────────────────────────────
    # FLIR ONE Pro 촬영 이미지 저장 폴더
    IMAGE_DIR = r"C:\Users\user\Documents\flir_one_pro_images"

    # ── 온도 범위 ────────────────────────────────────────────────────────────
    # FLIR ONE Pro 구간 1: -20°C ~ 120°C
    # [권장] FLIR ONE 앱 또는 FLIR Tools에서 확인 후 입력.
    #        Radiometric JPEG → "Export as CSV" 로 절대온도 직접 추출 가능.
    T_MIN = None    # 예: -20.0  (None=자동)
    T_MAX = None    # 예:  60.0  (None=자동)

    # 컬러맵: FLIR ONE Pro 기본값 = "ironbow"
    COLORMAP = DEFAULT_COLORMAP   # = "ironbow"

    # 카메라 ~ 도로 포장면 촬영 거리 [m] (픽셀 물리 크기 자동 계산에 사용)
    DISTANCE_M = 1.0

    # 처리할 최대 이미지 수 (None = 전체)
    MAX_IMAGES = None

    # 왜곡 보정 (FLIR ONE Pro NETD 70mK 기반 파라미터 자동 적용)
    APPLY_CORRECTION = True
    # ──────────────────────────────────────────────────────────────────────────

    print(f"카메라: {CAMERA_MODEL}  |  열 해상도: {ORIG_WIDTH}×{ORIG_HEIGHT}")
    print(f"컬러맵: {COLORMAP}  |  FEM 격자: {FEM_COLS}×{FEM_ROWS}")
    print(f"촬영거리: {DISTANCE_M:.1f}m")
    print(f"온도 범위: {T_MIN if T_MIN else _AUTO_T_MIN}°C ~ "
          f"{T_MAX if T_MAX else _AUTO_T_MAX}°C")
    print()

    if not os.path.isdir(IMAGE_DIR):
        print(f"⚠ 경로를 찾을 수 없습니다: {IMAGE_DIR}")
        print("  IMAGE_DIR 변수를 실제 경로로 수정 후 재실행하세요.")
        print()
        print("  [FLIR ONE Pro 권장 워크플로]")
        print("  1. FLIR ONE 앱에서 Radiometric JPEG 촬영")
        print("  2. FLIR Tools → File → Export → CSV (절대온도 직접 추출)")
        print("  3. T_MIN / T_MAX 입력 없이 CSV 로드 가능")
    else:
        loader = ThermalImageLoader(
            image_dir   = IMAGE_DIR,
            T_min       = T_MIN,
            T_max       = T_MAX,
            colormap    = COLORMAP,
            target_rows = FEM_ROWS,
            target_cols = FEM_COLS,
            seq_range   = None,
            distance_m  = DISTANCE_M,
        )

        results = run_batch_pipeline(
            loader,
            max_images       = MAX_IMAGES,
            apply_correction = APPLY_CORRECTION,
            save_figs        = True,
            output_dir       = OUTPUT_DIR,
        )
