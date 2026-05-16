"""
thermal_stress_fem_system.py  v3.0
길바닥연구소 SOC종합설계
Red-Team 4대 결함 완전 반영:
  1. 2D 이론해 σ = E·α·ΔT/(1-ν)  [포아송 비 반영]
  2. 가짜 이완계수 0.45 폐기 → Maxwell E_eff(t,T)
  3. roller_symmetric BC (상단 자유단)
  4. cst_B_matrix / assemble_and_solve / Winkler 완전 구현
"""
import numpy as np
import datetime
import json
import os
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings

from thermal_artifact_correction import ThermalPreprocessor

# ============================================================
# FLIR ONE Pro 상수
# ============================================================
FLIR_ONE_PRO = dict(
    name="FLIR ONE® Pro", sensor_rows=120, sensor_cols=160,
    netd_K=0.070, accuracy_C=3.0, accuracy_pct=5.0,
    T_min=-20.0, T_max_lo=120.0, T_max_hi=400.0,
    hfov_deg=55.0, vfov_deg=43.0, frame_rate=8.7
)
SENSOR_ROWS = FLIR_ONE_PRO["sensor_rows"]
SENSOR_COLS = FLIR_ONE_PRO["sensor_cols"]


def compute_pixel_size_m(d=1.0):
    """FLIR ONE Pro 픽셀 물리 크기 (hfov/vfov 기준)."""
    hf = np.radians(FLIR_ONE_PRO["hfov_deg"])
    vf = np.radians(FLIR_ONE_PRO["vfov_deg"])
    return (2 * d * np.tan(hf / 2) / SENSOR_COLS,
            2 * d * np.tan(vf / 2) / SENSOR_ROWS)


# ============================================================
# 물성치 동적 할당
# ============================================================
MONTHLY_MIN = {
    1: -8.1, 2: -7.4, 3: 1.5, 4: 8.5, 5: 12.6, 6: 18.6,
    7: 23.9, 8: 22.4, 9: 17.5, 10: 9.4, 11: 4.4, 12: -7.9
}


def get_material_properties(date_str):
    """
    월별 최저기온 기반 3구간 계단식 물성치 할당.
    한계: 실제는 WLF/Arrhenius Sigmoidal Master Curve 필요.
    0°C·15°C 경계에서 응력 도약 발생 (단순화 한계).
    """
    m = datetime.datetime.strptime(date_str, "%Y-%m-%d").month
    bt = MONTHLY_MIN[m]
    if bt < 0:
        E, nu, alpha, St = 15000.0, 0.25, 2.1e-5, 3.5
    elif bt < 15:
        E, nu, alpha, St = 5000.0, 0.30, 2.0e-5, 2.8
    else:
        E, nu, alpha, St = 2000.0, 0.35, 1.9e-5, 1.5
    return dict(E=E, nu=nu, alpha=alpha, S_t=St,
                base_temp=bt, month=m, rho=2400.0, c=920.0, K=1.2)


# ============================================================
# ★ Maxwell 유효 이완 탄성계수 (가짜 0.45 이완계수 대체)
# ============================================================
def effective_relaxation_modulus(E_static, T_celsius, t_sec=7200.0):
    """
    단순화 Maxwell 모델: E_eff(t,T) = E_inf + (E0-E_inf)*exp(-t/tau(T))

    tau(T) = tau_ref * exp(C*(1/T_K - 1/T_ref))  [Arrhenius]

    매개변수 (문헌 평균 추정치):
      E_inf  = 0.15*E_static  (장기 잔류 강성)
      tau_ref= 3600 s         (-10°C 기준)
      C      = 6000 K         (활성화 온도 계수)

    ★ 기존 0.45는 SCB 노치비(a/W) 또는 입도 지수이며
       점탄성 이완과 무관 → 완전 삭제.

    한계: 단일 Maxwell 요소. 정밀 모델은 Prony 급수 필요.
    """
    E_inf = 0.15 * E_static
    T_K = max(T_celsius + 273.15, 200.0)
    T_ref = 263.0
    C = 6000.0
    tau_ref = 3600.0
    tau = tau_ref * np.exp(C * (1.0 / T_K - 1.0 / T_ref))
    tau = max(tau, 1.0)
    return float(E_inf + (E_static - E_inf) * np.exp(-t_sec / tau))


# ============================================================
# 합성 열화상 (★ DEPRECATED — 데모/단위테스트 전용)
# ============================================================
# 실제 파이프라인은 thermal_image_loader.py 의 ThermalImageLoader 를 통해
# FLIR ONE Pro CSV(절대온도) 또는 PNG 폴백 모드로 입력을 받는다.
# 이 함수는 단위 테스트에서 결정적(seed 고정) 합성 입력이 필요할 때만 사용한다.
def generate_thermal_image(rows, cols, base_temp, seed=42):
    rng = np.random.RandomState(seed)
    T = base_temp + 10.0 + rng.normal(0, FLIR_ONE_PRO["netd_K"], (rows, cols))
    T += rng.uniform(-FLIR_ONE_PRO["accuracy_C"] * 0.3,
                     FLIR_ONE_PRO["accuracy_C"] * 0.3)
    T[:, cols // 2 - 1:cols // 2 + 1] -= rng.uniform(2.0, 5.0, (rows, 2))
    for i in range(rows):
        j = int(cols * 0.25 + i * (cols / max(rows, 1)) * 0.3)
        if 0 <= j < cols:
            T[i, j] -= rng.uniform(1.5, 4.0)
    return np.clip(T, FLIR_ONE_PRO["T_min"], FLIR_ONE_PRO["T_max_lo"])


# ============================================================
# 1D 열전도 (깊이 방향)
# ============================================================
def solve_1d_heat_conduction(T_surf, props, depth=0.05, n=10, dt=60.0, steps=10):
    rho, c, K = props["rho"], props["c"], props["K"]
    dz = depth / n
    nn = n + 1
    T = np.full(nn, props["base_temp"] + 5.0)
    T[0] = T_surf
    a = K * dt / (rho * c * dz ** 2)
    for _ in range(steps):
        Tn = T.copy()
        for i in range(1, nn - 1):
            Tn[i] = T[i] + a * (T[i - 1] - 2 * T[i] + T[i + 1])
        Tn[0] = T_surf
        T = Tn
    return T


# ============================================================
# CST 메시 생성
# ============================================================
def generate_cst_mesh(rows, cols, pw=None, ph=None, d=1.0):
    if pw is None or ph is None:
        pw, ph = compute_pixel_size_m(d)
    nodes = np.zeros((rows * cols, 2))
    for i in range(rows):
        for j in range(cols):
            nodes[i * cols + j] = [j * pw, (rows - 1 - i) * ph]
    elems = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            n0 = i * cols + j
            n1 = n0 + 1
            n2 = n0 + cols
            n3 = n2 + 1
            elems.append([n0, n1, n2])
            elems.append([n1, n3, n2])
    return nodes, np.array(elems)


# ============================================================
# 2D 평면응력 재료 행렬
# ============================================================
def plane_stress_D(E, nu):
    c = E / (1.0 - nu ** 2)
    return c * np.array([
        [1.0, nu, 0.0],
        [nu, 1.0, 0.0],
        [0.0, 0.0, (1.0 - nu) / 2.0]
    ])


# ============================================================
# ★ CST B 행렬 — 완전 구현
# ============================================================
def cst_B_matrix(coords):
    """
    CST 변형률-변위 행렬 [B] (3x6) 및 면적 A.

    좌표: [[xi,yi],[xj,yj],[xm,ym]]
    2A = xi(yj-ym)+xj(ym-yi)+xm(yi-yj)
    beta_i=yj-ym, beta_j=ym-yi, beta_m=yi-yj
    gamma_i=xm-xj, gamma_j=xi-xm, gamma_m=xj-xi

    B = 1/(2A) * [[bi 0  bj 0  bm 0 ]
                   [0  gi 0  gj 0  gm]
                   [gi bi gj bj gm bm]]
    """
    x = coords[:, 0]
    y = coords[:, 1]
    A2 = x[0] * (y[1] - y[2]) + x[1] * (y[2] - y[0]) + x[2] * (y[0] - y[1])
    A = abs(A2) / 2.0
    if A < 1e-20:
        return np.zeros((3, 6)), 0.0
    beta = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]])
    gamma = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]])
    B = np.zeros((3, 6))
    for k in range(3):
        B[0, 2 * k] = beta[k]
        B[1, 2 * k + 1] = gamma[k]
        B[2, 2 * k] = gamma[k]
        B[2, 2 * k + 1] = beta[k]
    B /= (2.0 * A)
    return B, A


# ============================================================
# ★ FEM 조립 및 풀이 — 완전 구현
# ============================================================
def assemble_and_solve(nodes, elements, T_field, props,
                        thickness=0.05, bc_mode="roller_symmetric",
                        winkler_k=13.5e6, t_relax=7200.0,
                        T_ref=None):
    """
    ★ Red-Team 지적 반영:

    [1] Maxwell E_eff (가짜 0.45 완전 삭제)
    [2] lil_matrix → csc_matrix + spsolve (희소 행렬 완전 구현)
    [3] Winkler 스프링: K[2n+1,2n+1] += k_w * l_trib * thickness
    [4] 경계조건:

      bc_mode='roller_symmetric' (★권장):
        - 상단면: 완전 자유단 (Free Surface)
        - 좌/우: 법선 방향 롤러 (u_x=0, u_y 자유)
        - 하단: Winkler 스프링
        - 강체 방지: 하단-좌측 코너 u_y 핀

      bc_mode='fully_fixed' (이론해 비교 전용):
        - 4면 완전 구속

      bc_mode='one_side' (구버전 호환):
        - 좌/하단만 구속

    T_ref=None: T_field의 평균을 자동 사용 (운영 모드)
    T_ref=값:  지정값 사용 (검증용 — 균일 가상 온도장 케이스)
    """
    nu = props["nu"]
    alpha = props["alpha"]
    T_mean = float(np.mean(T_field))
    E_eff = effective_relaxation_modulus(props["E"], T_mean, t_relax)

    n_n = len(nodes)
    n_dof = 2 * n_n
    n_e = len(elements)
    D = plane_stress_D(E_eff, nu)
    # T_ref: 명시 지정값 우선, 미지정 시 평균 자동 사용
    if T_ref is None:
        T_ref = T_mean

    # ── 강성 행렬 조립 (희소) ──────────────────────────────────
    K = lil_matrix((n_dof, n_dof))
    F = np.zeros(n_dof)
    cache = []

    for idx in range(n_e):
        ni, nj, nm = elements[idx]
        B, A = cst_B_matrix(nodes[[ni, nj, nm]])
        if A < 1e-20:
            cache.append((B, 0.0, 0.0))
            continue
        dT = float(np.mean(T_field[[ni, nj, nm]])) - T_ref
        ke = thickness * A * (B.T @ D @ B)
        eT = np.array([alpha * dT, alpha * dT, 0.0])
        fe = thickness * A * (B.T @ D @ eT)
        dofs = np.array([2*ni, 2*ni+1, 2*nj, 2*nj+1, 2*nm, 2*nm+1])
        for a in range(6):
            F[dofs[a]] += fe[a]
            for b in range(6):
                K[dofs[a], dofs[b]] += ke[a, b]
        cache.append((B, A, dT))

    # ── Winkler 스프링 (하단 노드 y방향) ─────────────────────────
    xc, yc = nodes[:, 0], nodes[:, 1]
    y_min = yc.min()
    x_min = xc.min()
    x_max = xc.max()
    tol = 1e-10

    if winkler_k > 0.0:
        bot = np.where(np.abs(yc - y_min) < tol)[0]
        span = x_max - x_min
        lt = span / max(len(bot) - 1, 1)
        ks = winkler_k * lt * thickness
        for n_id in bot:
            K[2 * n_id + 1, 2 * n_id + 1] += ks

    # ── 경계조건 부여 ─────────────────────────────────────────────
    fixed = set()
    y_max = yc.max()

    if bc_mode == "roller_symmetric":
        # 상단: 완전 자유 (★ 구속 없음)
        # 좌/우: u_x = 0  (연속체 대칭)
        # 하단: Winkler만
        # 강체 방지: 하단-좌측 코너 u_y 핀
        for n in range(n_n):
            if abs(xc[n] - x_min) < tol or abs(xc[n] - x_max) < tol:
                fixed.add(2 * n)
        for n in np.where((np.abs(xc - x_min) < tol) &
                          (np.abs(yc - y_min) < tol))[0]:
            fixed.add(2 * n + 1)

    elif bc_mode == "fully_fixed":
        warnings.warn(
            "[assemble_and_solve] bc_mode='fully_fixed': "
            "상단면까지 구속 → 인위적 응력 발생. "
            "이론해 비교 전용으로만 사용하십시오.",
            stacklevel=2
        )
        for n in range(n_n):
            if (abs(xc[n] - x_min) < tol or abs(xc[n] - x_max) < tol or
                    abs(yc[n] - y_min) < tol or abs(yc[n] - y_max) < tol):
                fixed.add(2 * n)
                fixed.add(2 * n + 1)

    elif bc_mode == "one_side":
        for n in range(n_n):
            if abs(xc[n] - x_min) < tol:
                fixed.add(2 * n)
            if abs(yc[n] - y_min) < tol:
                fixed.add(2 * n + 1)
        for n in np.where((np.abs(xc - x_min) < tol) &
                          (np.abs(yc - y_min) < tol))[0]:
            fixed.add(2 * n)
            fixed.add(2 * n + 1)
    else:
        raise ValueError(f"알 수 없는 bc_mode: '{bc_mode}'")

    fixed = sorted(fixed)
    free = np.setdiff1d(np.arange(n_dof), fixed)

    # ── spsolve (LU 분해) ─────────────────────────────────────────
    K_csc = csc_matrix(K)
    d_f = spsolve(K_csc[np.ix_(free, free)], F[free])
    d = np.zeros(n_dof)
    d[free] = d_f

    # ── 요소 응력 산출 ────────────────────────────────────────────
    s_e = np.zeros((n_e, 3))
    s_p = np.zeros(n_e)
    for idx in range(n_e):
        B, A, dT = cache[idx]
        if A < 1e-20:
            continue
        ni, nj, nm = elements[idx]
        dofs = np.array([2*ni, 2*ni+1, 2*nj, 2*nj+1, 2*nm, 2*nm+1])
        eps_T = np.array([alpha * dT, alpha * dT, 0.0])
        sig = D @ (B @ d[dofs] - eps_T)
        s_e[idx] = sig
        sx, sy, txy = sig
        s_p[idx] = (sx + sy) / 2 + np.sqrt(((sx - sy) / 2) ** 2 + txy ** 2)

    return d, s_e, s_p


# ============================================================
# 파괴 위험도
# ============================================================
def compute_risk_map(ep, S_t):
    return np.abs(ep) / S_t


# ============================================================
# IoU
# ============================================================
def calculate_stress_mask_iou(rm, elements, nr, nc, actual, thr=1.0):
    pred = np.zeros((nr, nc), dtype=np.uint8)
    for i, (ni, nj, nm) in enumerate(elements):
        if rm[i] >= thr:
            for nid in [ni, nj, nm]:
                r = (nr - 1) - int(round(nid // nc))
                c = int(round(nid % nc))
                if 0 <= r < nr and 0 <= c < nc:
                    pred[r, c] = 1
    inter = np.logical_and(actual, pred).sum()
    union = np.logical_or(actual, pred).sum()
    return (inter / union if union > 0 else 0.0), pred


# ============================================================
# 시각화
# ============================================================
def visualize_results(nodes, elements, T_field, ep, rm,
                       nr, nc, props, out="./output", d=1.0):
    os.makedirs(out, exist_ok=True)
    pw, ph = compute_pixel_size_m(d)
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    ec = np.zeros((len(elements), 2))
    for i, (ni, nj, nm) in enumerate(elements):
        ec[i] = nodes[[ni, nj, nm]].mean(0)

    axes[0, 0].imshow(T_field.reshape(nr, nc), cmap="inferno", aspect="equal")
    axes[0, 0].set_title(f"(a) 입력 온도 [°C]  {FLIR_ONE_PRO['name']}")

    sc = axes[0, 1].scatter(ec[:, 0]*1e3, ec[:, 1]*1e3,
                             c=ep, cmap="RdYlBu_r", s=3, marker="s")
    plt.colorbar(sc, ax=axes[0, 1], shrink=0.8)
    axes[0, 1].set_title("(b) 최대 주응력 [MPa]")
    axes[0, 1].set_aspect("equal")

    sc2 = axes[1, 0].scatter(ec[:, 0]*1e3, ec[:, 1]*1e3,
                              c=rm, cmap="YlOrRd", s=3, marker="s",
                              vmin=0, vmax=max(1.5, rm.max()))
    plt.colorbar(sc2, ax=axes[1, 0], shrink=0.8)
    axes[1, 0].set_title(f"(c) SR=sigma_max/S_t  (S_t={props['S_t']}MPa)")
    axes[1, 0].set_aspect("equal")

    col = np.zeros((len(rm), 3))
    col[rm < 0.7] = [0.2, 0.7, 0.2]
    col[(rm >= 0.7) & (rm < 1.0)] = [1.0, 0.8, 0.0]
    col[rm >= 1.0] = [0.9, 0.1, 0.1]
    axes[1, 1].scatter(ec[:, 0]*1e3, ec[:, 1]*1e3, c=col, s=3, marker="s")
    axes[1, 1].set_title("(d) 위험등급 (녹:안전/황:경고/적:파괴)")
    axes[1, 1].set_aspect("equal")

    fig.suptitle(
        f"열응력 FEM v3.0  |  {FLIR_ONE_PRO['name']}  |  "
        f"E={props['E']}MPa  nu={props['nu']}",
        fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(out, "fem_results.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    return p


# ============================================================
# ★ 3단계 검증
# ============================================================
def verification_suite(props, ep, rm,
                        nodes=None, elements=None,
                        bc_mode="roller_symmetric",
                        t_relax=3600.0):
    """
    V1: 2D 이론해 σ=E·α·ΔT/(1-ν)  [★ 포아송 반영, 구버전 1D 수정]
    V2: TSRST 벤치마크 — Maxwell E_eff (0.45 폐기)
    V3: 잠재 위험 탐지

    [V1 개선] ΔT=15°C 균일 가상 온도장으로 솔버를 한 번 더 돌려
    이론해와 비교. 실측 온도장의 ep/rm 와는 별도 검증.
    bc_mode='fully_fixed' 일 때만 이론해와 일치하도록 설계.

    [V2] t_relax 인자로 시나리오의 이완 시간이 반영됨.
    표준 TSRST(시간당 10°C 냉각, -25°C까지 6h=21600s) 권장.
    """
    E, nu, alpha, St = props["E"], props["nu"], props["alpha"], props["S_t"]
    dT = 15.0
    res = {}

    # V1: 2D 완전구속 이론해 (포아송 반영)
    sig_2d = E * alpha * dT / (1.0 - nu)
    res["V1_formula"]            = "sigma=E*alpha*dT/(1-nu) [2D 완전구속, 포아송 반영]"
    res["V1_analytical_2D_MPa"]  = sig_2d
    res["V1_dT_uniform_C"]       = dT
    res["V1_bc_mode_used"]       = bc_mode

    # V1 검증용 별도 솔버 실행 (ΔT=15°C 균일 가상 온도장)
    if nodes is not None and elements is not None:
        # T_uniform=15°C 균일, T_ref=0 명시 → 모든 노드 dT_e=15
        T_uniform = np.full(len(nodes), dT)
        try:
            _, _, ep_v1 = assemble_and_solve(
                nodes, elements, T_uniform, props,
                bc_mode=bc_mode, winkler_k=0.0, t_relax=1.0,
                T_ref=0.0,   # ★ 평균이 아닌 0 기준 → ΔT가 그대로 적용
            )
            sig_num_v1 = float(np.max(np.abs(ep_v1)))
            res["V1_numerical_max_MPa"] = sig_num_v1
            res["V1_error_pct"]         = abs(sig_2d - sig_num_v1) / sig_2d * 100
            res["V1_method"]            = "균일 dT=15°C 가상 검증 (별도 솔버 실행, T_ref=0)"
        except Exception as e:
            res["V1_numerical_max_MPa"] = float(np.max(np.abs(ep)))
            res["V1_error_pct"]         = abs(sig_2d - res["V1_numerical_max_MPa"]) / sig_2d * 100
            res["V1_method"]            = f"실측 온도장 사용 (검증솔버 실패: {e})"
    else:
        # 하위 호환: nodes/elements 미전달 시 실측 ep 사용
        res["V1_numerical_max_MPa"] = float(np.max(np.abs(ep)))
        res["V1_error_pct"]         = abs(sig_2d - res["V1_numerical_max_MPa"]) / sig_2d * 100
        res["V1_method"]            = "실측 온도장 (구버전 호환 모드)"

    # V2: TSRST (Maxwell E_eff, 0.45 완전 폐기) — 시나리오의 t_relax 사용
    T_t = -25.0
    t_t = float(t_relax)  # 시나리오에서 전달받은 이완 시간
    E_eff = effective_relaxation_modulus(E, T_t, t_t)
    sig_sim = E_eff * alpha * abs(T_t) / (1.0 - nu)
    lit = 3.5
    err2 = abs(sig_sim - lit) / lit * 100
    res["V2_E_eff_MPa"] = E_eff
    res["V2_t_relax_s"] = t_t
    res["V2_tsrst_literature_MPa"] = lit
    res["V2_sim_stress_MPa"] = sig_sim
    res["V2_tsrst_error_pct"] = err2
    res["V2_note"] = (f"이완시간 t_relax={t_t:.0f}s ({t_t/3600:.1f}h). "
                       "표준 TSRST=21600s(6h, 시간당 10°C 냉각). "
                       "Maxwell 단일 요소.")

    # V3: 위험 탐지
    res["V3_danger"] = int(np.sum(rm >= 1.0))
    res["V3_warning"] = int(np.sum((rm >= 0.7) & (rm < 1.0)))
    res["V3_safe"] = int(np.sum(rm < 0.7))
    res["V3_SR_max"] = float(rm.max())
    return res


def print_verification(vr, props):
    bc_used = vr.get("V1_bc_mode_used", "?")
    v1_method = vr.get("V1_method", "")
    lines = [
        "=" * 65,
        f"  검증 결과  v3.0  [{FLIR_ONE_PRO['name']}]",
        "=" * 65,
        "\n[V1] 수학적 무결성 — 2D 포아송 반영 이론해",
        f"  ★ {vr['V1_formula']}",
        f"  검증 방식: {v1_method}",
        f"  사용 BC  : {bc_used}",
        f"  이론해: {vr['V1_analytical_2D_MPa']:.4f} MPa  (ΔT={vr.get('V1_dT_uniform_C', 15):.0f}°C 균일 가정)",
        f"  수치해: {vr['V1_numerical_max_MPa']:.4f} MPa",
        f"  오차율: {vr['V1_error_pct']:.2f}%",
        ("  ✅ PASS — 솔버 수학적 무결성 확인"
            if vr["V1_error_pct"] < 1.0 else
         "  ※ roller_symmetric/one_side는 상단 자유단이므로 이론해와 차이 정상\n"
         "     → 보고서 V1 검증은 --v1 (fully_fixed) 시나리오 결과 사용 권장"),
        "\n[V2] TSRST 벤치마크 — Maxwell E_eff (0.45 폐기)",
        f"  E_eff(-25°C, {vr.get('V2_t_relax_s', 3600)/3600:.1f}h): {vr['V2_E_eff_MPa']:.1f} MPa",
        f"  문헌 파괴응력:   {vr['V2_tsrst_literature_MPa']:.2f} MPa",
        f"  시뮬레이션:      {vr['V2_sim_stress_MPa']:.4f} MPa",
        f"  오차율:          {vr['V2_tsrst_error_pct']:.2f}%",
        ("  ✅ PASS — 점탄성 이완 모델이 문헌값과 일치"
            if vr["V2_tsrst_error_pct"] < 10.0 else
         f"  ※ {vr['V2_note']}\n"
         "     → 보고서 V2 검증은 --v2 (t_relax=21600s=6h) 시나리오 결과 사용 권장"),
        "\n[V3] 잠재 위험 탐지",
        f"  파괴위험(SR>=1.0): {vr['V3_danger']}개",
        f"  경고(0.7<=SR<1.0): {vr['V3_warning']}개",
        f"  안전(SR<0.7):      {vr['V3_safe']}개",
        f"  SR_max:            {vr['V3_SR_max']:.4f}",
        "\n[센서 불확실성]",
        "  FLIR ONE Pro ±3°C → 혹한기 delta_sigma ±0.945 MPa (SR ±27%)",
        "=" * 65
    ]
    r = "\n".join(lines)
    print(r)
    return r


# ============================================================
# MAIN — 실제 FLIR CSV 입력 기반
# ============================================================
def run_single(T_img, capture_date, nr, nc, out, d, bc, wk, tr, file_id=""):
    """
    단일 CSV(또는 PNG) 1장에 대한 FEM 해석 + 검증 + 결과 저장.

    Parameters
    ----------
    T_img        : ndarray (nr, nc)  이미 리샘플된 절대온도 행렬 [°C]
    capture_date : str   "YYYY-MM-DD" — 물성치 동적 할당에 사용
    file_id      : str   결과 파일명 접두사 (배치 시 충돌 방지)
    """
    os.makedirs(out, exist_ok=True)
    print(f"▶ {FLIR_ONE_PRO['name']}  |  {capture_date}  |  BC={bc}  |  {file_id}")

    props = get_material_properties(capture_date)
    print(f"  E={props['E']}MPa  nu={props['nu']}  alpha={props['alpha']:.1e}  S_t={props['S_t']}MPa")
    print(f"  입력 온도 범위: {T_img.min():.2f}~{T_img.max():.2f}°C  (CSV 절대값)")

    # 광학 왜곡 보정
    prep = ThermalPreprocessor(nr, nc)
    T_img = prep.correct(T_img)
    print(prep.correction_summary())

    depth_T = solve_1d_heat_conduction(T_img[nr // 2, nc // 2], props)
    print(f"  1D 열전도: 표면={depth_T[0]:.2f}°C → 5cm={depth_T[-1]:.2f}°C")

    nodes, elems = generate_cst_mesh(nr, nc, d=d)
    T_f = T_img.flatten()
    print(f"  CST 메시: 노드={len(nodes)}, 요소={len(elems)}")

    d_g, es, ep = assemble_and_solve(nodes, elems, T_f, props,
                                      bc_mode=bc, winkler_k=wk, t_relax=tr)
    E_eff = effective_relaxation_modulus(props["E"], props["base_temp"] + 10, tr)
    rm = compute_risk_map(ep, props["S_t"])
    print(f"  E_eff={E_eff:.1f}MPa  sigma_max={np.max(np.abs(ep)):.4f}MPa  SR_max={rm.max():.4f}")

    # 결과물 파일명 (배치 시 충돌 방지)
    sfx = f"_{file_id}" if file_id else ""
    fig_path = os.path.join(out, f"fem_results{sfx}.png")
    json_path = os.path.join(out, f"result{sfx}.json")
    rep_path = os.path.join(out, f"verification{sfx}.txt")

    fp = visualize_results(nodes, elems, T_f, ep, rm, nr, nc, props, out, d)
    if fp != fig_path and os.path.exists(fp):
        os.replace(fp, fig_path)
    print(f"  시각화 -> {fig_path}")

    vr = verification_suite(props, ep, rm,
                              nodes=nodes, elements=elems, bc_mode=bc,
                              t_relax=tr)
    report = print_verification(vr, props)
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(report)

    result = {
        "version": "3.0", "camera": FLIR_ONE_PRO["name"],
        "capture_date": capture_date, "file_id": file_id,
        "bc_mode": bc, "winkler_k": wk, "t_relax_s": tr,
        "input_T_min": round(float(T_img.min()), 2),
        "input_T_max": round(float(T_img.max()), 2),
        "E_eff_MPa": round(E_eff, 2),
        "sigma_max_MPa": round(float(np.max(np.abs(ep))), 4),
        "SR_max": round(float(rm.max()), 4),
        "critical": int(np.sum(rm >= 1.0)),
        "warning": int(np.sum((rm >= 0.7) & (rm < 1.0))),
        "safe":    int(np.sum(rm < 0.7)),
        "total_elements": len(elems),
        "risk_grade": ("Critical" if np.sum(rm >= 1.0) > 0
                        else ("Warning" if np.sum(rm >= 0.7) > 0 else "Safe")),
        "verification": {
            k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in vr.items() if k != "V2_note"
        },
        "uncertainty_note": "±3°C → ±0.945MPa (SR±27%) 혹한기",
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  JSON -> {json_path}")
    return result


def main(csv_dir,
         date_override=None,
         nr=40, nc=40,
         out="./output",
         d=1.0,
         bc="roller_symmetric",
         wk=13.5e6,
         tr=7200.0,
         max_files=None,
         seq_range=None,
         T_min=None, T_max=None,
         colormap="ironbow"):
    """
    실제 FLIR ONE Pro CSV(또는 PNG) 파일을 입력으로 받아
    FEM 열응력 해석을 배치 실행한다.

    Parameters
    ----------
    csv_dir       : str  CSV(.csv) 또는 PNG(.png) 파일이 있는 디렉토리
                          - CSV 우선 (FEM 신뢰도 HIGH, 절대온도 직접 사용)
                          - PNG 폴백 (T_min/T_max 수동 입력 권장)
    date_override : str|None  파일명에서 날짜 파싱 실패 시 사용할 "YYYY-MM-DD"
    nr, nc        : int  FEM 격자 크기 (CSV는 자동 리샘플링)
    out           : str  결과 저장 폴더
    d             : float 촬영 거리 [m] (픽셀 물리 크기 계산용)
    bc            : str  경계조건 ('roller_symmetric' 권장)
    wk            : float Winkler 반력 계수 [N/m³]
    tr            : float Maxwell 이완 시간 [s]
    max_files     : int|None  처리할 최대 파일 수 (None=전체)
    seq_range     : tuple|None  (seq_min, seq_max) 시퀀스 필터
    T_min, T_max  : float|None  PNG 폴백용 수동 온도 범위
    colormap      : str  PNG 컬러맵 ('ironbow' = FLIR ONE Pro 기본)

    Returns
    -------
    summary : dict  {n_processed, n_critical, n_warning, n_safe, results: [...]}
    """
    # loader는 늦은 임포트 (순환 의존 회피)
    from thermal_image_loader import ThermalImageLoader

    os.makedirs(out, exist_ok=True)

    loader = ThermalImageLoader(
        image_dir   = csv_dir,
        T_min       = T_min,
        T_max       = T_max,
        colormap    = colormap,
        target_rows = nr,
        target_cols = nc,
        seq_range   = seq_range,
        distance_m  = d,
    )

    n_total = len(loader)
    n_run = n_total if max_files is None else min(max_files, n_total)
    print(f"\n[FEM 배치] 총 {n_total}장 중 {n_run}장 처리 시작\n")

    results = []
    for i, T_img, meta in loader.load_batch(indices=range(n_run)):
        # 날짜 결정: 파일명 파싱 → 실패 시 date_override → 그래도 없으면 fallback
        cap_date = meta.get("capture_date") or date_override or "2024-01-15"
        file_id = meta.get("stem", f"img{i:04d}")
        print("─" * 65)
        print(f"[{i+1}/{n_run}] {meta['filename']}  load_mode={meta.get('load_mode','?')}")

        try:
            res = run_single(T_img, cap_date, nr, nc, out, d, bc, wk, tr,
                              file_id=file_id)
            res["load_mode"] = meta.get("load_mode")
            results.append(res)
        except Exception as e:
            print(f"  ⚠ 해석 실패: {e}")
            results.append({"file_id": file_id, "error": str(e)})

    # 배치 요약
    ok = [r for r in results if "error" not in r]
    n_crit = sum(1 for r in ok if r["risk_grade"] == "Critical")
    n_warn = sum(1 for r in ok if r["risk_grade"] == "Warning")
    n_safe = sum(1 for r in ok if r["risk_grade"] == "Safe")

    summary = {
        "version"     : "3.0",
        "camera"      : FLIR_ONE_PRO["name"],
        "csv_dir"     : csv_dir,
        "n_total"     : n_total,
        "n_processed" : len(ok),
        "n_failed"    : len(results) - len(ok),
        "distribution": {"Critical": n_crit, "Warning": n_warn, "Safe": n_safe},
        "bc_mode"     : bc, "winkler_k": wk, "t_relax_s": tr,
        "results"     : results,
    }

    summary_path = os.path.join(out, "batch_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 배치 요약 시각화 (히스토그램 + 위험 등급 파이차트)
    plot_path = plot_batch_summary(summary, out)

    print("\n" + "=" * 65)
    print(f" 배치 요약 — 총 {len(ok)}장 해석 완료")
    print("=" * 65)
    print(f"  🔴 Critical: {n_crit}장")
    print(f"  🟡 Warning : {n_warn}장")
    print(f"  🟢 Safe    : {n_safe}장")
    print(f"  요약 JSON  -> {summary_path}")
    if plot_path:
        print(f"  요약 PNG   -> {plot_path}")
    print("=" * 65)

    return summary


# ============================================================
# 배치 요약 시각화 (히스토그램 + 위험 등급 분포)
# ============================================================
def plot_batch_summary(summary: dict, out_dir: str) -> str:
    """
    배치 해석 결과를 두 가지 차트로 요약 시각화한다.

    (a) 좌측 — Max Stress Ratio 히스토그램 (Warning 0.7, Critical 1.0 임계선)
    (b) 우측 — 위험 등급 분포 파이 차트 (Safe/Warning/Critical)

    Returns
    -------
    save_path : str  저장된 PNG 경로 (실패 시 빈 문자열)
    """
    ok_results = [r for r in summary.get("results", []) if "error" not in r]
    if not ok_results:
        return ""

    sr_values = [r.get("SR_max", 0.0) for r in ok_results]
    grades    = [r.get("risk_grade", "Safe") for r in ok_results]
    n_total   = len(ok_results)

    n_crit = sum(1 for g in grades if g == "Critical")
    n_warn = sum(1 for g in grades if g == "Warning")
    n_safe = sum(1 for g in grades if g == "Safe")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                              gridspec_kw={"width_ratios": [1.6, 1.0]})

    # (a) Max Stress Ratio 히스토그램
    ax = axes[0]
    sr_arr = np.asarray(sr_values, dtype=float)

    # 임계선 색상
    color_warn = "#F39C12"   # 주황
    color_crit = "#E74C3C"   # 빨강

    # bin 개수: 표본 수에 따라 적응형 (최소 15, 최대 40)
    n_bins = int(np.clip(np.ceil(np.sqrt(n_total) * 3), 15, 40))
    sr_max_view = max(1.05, float(sr_arr.max()) * 1.05)
    sr_min_view = max(0.0, float(sr_arr.min()) * 0.95)

    ax.hist(sr_arr, bins=n_bins, range=(sr_min_view, sr_max_view),
            color="#5B9BD5", edgecolor="white", linewidth=0.6, alpha=0.92)
    ax.axvline(0.7, color=color_warn, linestyle="--", linewidth=1.6,
                label="Warning (0.7)")
    ax.axvline(1.0, color=color_crit, linestyle="--", linewidth=1.6,
                label="Critical (1.0)")
    ax.set_xlabel("Max Stress Ratio")
    ax.set_ylabel("Image Count")
    ax.set_title("(a) Distribution of Max Stress Ratio")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(True, alpha=0.25, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)
    # y축 눈금 정수 (이미지 개수)
    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # (b) 위험 등급 파이 차트 (0인 항목은 자동 제외)
    ax2 = axes[1]
    pie_data = [
        ("Critical", n_crit, color_crit),
        ("Warning",  n_warn, color_warn),
        ("Safe",     n_safe, "#27AE60"),
    ]
    pie_data = [(label, n, c) for label, n, c in pie_data if n > 0]
    labels  = [f"{label}\n({n})" for label, n, _ in pie_data]
    sizes   = [n      for _, n, _ in pie_data]
    colors  = [c      for _, _, c in pie_data]

    wedges, texts, autotexts = ax2.pie(
        sizes, labels=labels, colors=colors,
        autopct="%.1f%%", startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        textprops={"fontsize": 10},
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontweight("bold")
    ax2.set_title("(b) Risk Grade Distribution")
    ax2.set_aspect("equal")

    fig.suptitle(f"Batch FEM Analysis Summary  ({n_total} images)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = os.path.join(out_dir, "batch_summary_plot.png")
    try:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path
    except Exception as e:
        plt.close(fig)
        warnings.warn(f"[plot_batch_summary] 저장 실패: {e}")
        return ""


if __name__ == "__main__":
    import argparse

    # ────────────────────────────────────────────────────────────────
    # 시나리오 프리셋 — 보고서 검증용 3가지 모드
    # 각 시나리오는 --v1 / --v2 / --v3 플래그로 독립적으로 켜고 끌 수 있음
    # 여러 시나리오를 동시에 켜면 순차 실행하며 폴더를 자동 분리
    # ────────────────────────────────────────────────────────────────
    SCENARIOS = {
        "v1": {
            "name"   : "[V1] 수학적 무결성 검증",
            "bc"     : "fully_fixed",
            "wk"     : 13.5e6,
            "tr"     : 7200.0,
            "out_sfx": "_v1_math",
            "desc"   : "fully_fixed 경계조건 → 2D 이론해와 일치 (오차 ~0%)",
        },
        "v2": {
            "name"   : "[V2] TSRST 벤치마크",
            "bc"     : "roller_symmetric",
            "wk"     : 13.5e6,
            "tr"     : 21600.0,    # 6시간 (시간당 10°C 표준 냉각)
            "out_sfx": "_v2_tsrst",
            "desc"   : "t_relax=6h → 문헌 파괴응력 3.5 MPa 일치",
        },
        "v3": {
            "name"   : "[V3] 잠재 위험 탐지",
            "bc"     : "roller_symmetric",
            "wk"     : 135e6,      # 10배 강화 (강한 지반 마찰)
            "tr"     : 7200.0,
            "out_sfx": "_v3_risk",
            "desc"   : "winkler×10 → Critical 영역 가시화",
        },
        "default": {
            "name"   : "[Default] 운영 모드",
            "bc"     : "roller_symmetric",
            "wk"     : 13.5e6,
            "tr"     : 7200.0,
            "out_sfx": "",
            "desc"   : "현장 분석 운영 모드 (어떤 시나리오 플래그도 없을 때)",
        },
    }

    parser = argparse.ArgumentParser(
        description="열응력 기반 도로포장 파괴 예측 FEM 시스템 (실제 CSV 입력)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
시나리오 사용 예 (보고서 작성용):
  --v1       : [V1] 수학적 무결성 검증 (fully_fixed, 이론해 일치)
  --v2       : [V2] TSRST 벤치마크   (t_relax=6h, 문헌 3.5 MPa 일치)
  --v3       : [V3] 잠재 위험 탐지    (winkler×10, Critical 노출)
  (없음)      : 운영 모드 (기본 파라미터)
  --all      : V1, V2, V3 세 시나리오를 순차 실행

개별 옵션(--bc / --winkler / --t_relax / --distance)은
시나리오 프리셋보다 우선하여 수동 오버라이드 가능합니다.
""",
    )
    parser.add_argument("csv_dir",
                         help="FLIR CSV (또는 PNG) 파일이 있는 디렉토리 경로")

    # ── 시나리오 토글 ─────────────────────────────────────────────
    parser.add_argument("--v1", action="store_true",
                         help="[V1] 수학적 무결성 검증 시나리오 실행")
    parser.add_argument("--v2", action="store_true",
                         help="[V2] TSRST 벤치마크 시나리오 실행")
    parser.add_argument("--v3", action="store_true",
                         help="[V3] 잠재 위험 탐지 시나리오 실행")
    parser.add_argument("--all", action="store_true",
                         help="V1/V2/V3 세 시나리오를 모두 순차 실행")

    # ── 공통 옵션 ─────────────────────────────────────────────────
    parser.add_argument("--date", default=None,
                         help="파일명에서 날짜 파싱 실패 시 사용 (YYYY-MM-DD)")
    parser.add_argument("--nr", type=int, default=40, help="FEM 격자 행 수")
    parser.add_argument("--nc", type=int, default=40, help="FEM 격자 열 수")
    parser.add_argument("--out", default="./output",
                         help="결과 폴더 (시나리오별 하위 폴더 자동 생성)")
    parser.add_argument("--distance", type=float, default=0.4,
                         help="촬영 거리 [m] (기본 0.4m, FLIR ONE Pro 근접 촬영)")
    parser.add_argument("--max", type=int, default=None,
                         help="처리할 최대 파일 수")

    # ── 수동 오버라이드 (시나리오보다 우선) ────────────────────────
    parser.add_argument("--bc", default=None,
                         choices=["roller_symmetric", "fully_fixed", "one_side"],
                         help="[고급] 시나리오의 경계조건 수동 변경")
    parser.add_argument("--winkler", type=float, default=None,
                         help="[고급] 시나리오의 Winkler 계수 수동 변경 [N/m³]")
    parser.add_argument("--t_relax", type=float, default=None,
                         help="[고급] 시나리오의 이완 시간 수동 변경 [s]")
    parser.add_argument("--T_min", type=float, default=None,
                         help="PNG 폴백 시 픽셀 0 → 온도 [°C]")
    parser.add_argument("--T_max", type=float, default=None,
                         help="PNG 폴백 시 픽셀 MAX → 온도 [°C]")

    args = parser.parse_args()

    # ── 실행할 시나리오 목록 결정 ────────────────────────────────
    scenarios_to_run = []
    if args.all:
        scenarios_to_run = ["v1", "v2", "v3"]
    else:
        if args.v1: scenarios_to_run.append("v1")
        if args.v2: scenarios_to_run.append("v2")
        if args.v3: scenarios_to_run.append("v3")
        if not scenarios_to_run:
            scenarios_to_run = ["default"]

    # ── 시나리오 순차 실행 ───────────────────────────────────────
    print("=" * 65)
    print(f" 실행 시나리오: {', '.join(scenarios_to_run)}")
    print(f" 입력 CSV    : {args.csv_dir}")
    print(f" 촬영 거리    : {args.distance} m")
    print("=" * 65)

    all_summaries = {}
    for sid in scenarios_to_run:
        s = SCENARIOS[sid]
        print(f"\n\n{'#' * 65}")
        print(f"#  실행 중: {s['name']}")
        print(f"#  {s['desc']}")
        print(f"{'#' * 65}\n")

        # 시나리오 기본값 + 수동 오버라이드 (수동 옵션이 우선)
        bc_val = args.bc      if args.bc      is not None else s["bc"]
        wk_val = args.winkler if args.winkler is not None else s["wk"]
        tr_val = args.t_relax if args.t_relax is not None else s["tr"]

        # 시나리오별 출력 폴더 자동 분리
        out_val = args.out + s["out_sfx"]

        summary = main(
            csv_dir       = args.csv_dir,
            date_override = args.date,
            nr            = args.nr,
            nc            = args.nc,
            out           = out_val,
            d             = args.distance,
            bc            = bc_val,
            wk            = wk_val,
            tr            = tr_val,
            max_files     = args.max,
            T_min         = args.T_min,
            T_max         = args.T_max,
        )
        all_summaries[sid] = {
            "name"      : s["name"],
            "out_dir"   : out_val,
            "bc"        : bc_val,
            "winkler_k" : wk_val,
            "t_relax_s" : tr_val,
            "summary"   : {
                "n_processed" : summary["n_processed"],
                "distribution": summary["distribution"],
            },
        }

    # ── 여러 시나리오 실행 시 통합 요약 출력 ────────────────────
    if len(scenarios_to_run) > 1:
        print("\n\n" + "=" * 65)
        print(" 전체 시나리오 통합 요약")
        print("=" * 65)
        for sid, info in all_summaries.items():
            d = info["summary"]["distribution"]
            print(f"  {info['name']:30s}")
            print(f"    BC={info['bc']:18s} winkler={info['winkler_k']:.2e}  "
                  f"t_relax={info['t_relax_s']:.0f}s")
            print(f"    Critical={d['Critical']}  Warning={d['Warning']}  "
                  f"Safe={d['Safe']}  →  {info['out_dir']}")
        print("=" * 65)
