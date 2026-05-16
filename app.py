"""
SOC 졸업설계 — 도로 포장 열응력 진단 시스템 v2.0
대시보드 UI (flet 기본 API만 사용 — 호환성 안전)
"""
import flet as ft
import requests
import os
import subprocess
import time
import sys
import threading

BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")
BACKEND_URL_PUBLIC = os.environ.get("BACKEND_URL_PUBLIC", BACKEND_URL)
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploaded_assets")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 한국도로공사 톤 컬러
COLOR_PRIMARY    = "#0D1B4C"
COLOR_SECONDARY  = "#1976D2"
COLOR_BG         = "#ECEFF4"
COLOR_CARD       = "#FFFFFF"
COLOR_DANGER     = "#C62828"
COLOR_WARNING    = "#F57C00"
COLOR_SAFE       = "#388E3C"
COLOR_BORDER     = "#CFD8DC"
COLOR_TEXT_MAIN  = "#212121"
COLOR_TEXT_SUB   = "#607D8B"


def main(page: ft.Page):
    # ── 백엔드 시작 ───────────────────────────────────────────
    def start_backend():
        try:
            requests.get(f"{BACKEND_URL}/", timeout=2)
            print("✅ 백엔드 이미 실행 중")
            return
        except requests.RequestException:
            pass
        base_path = os.path.dirname(os.path.abspath(__file__))
        print("🚀 백엔드 시작 중...")
        subprocess.Popen(
            [sys.executable, "main.py"], cwd=base_path,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )
        deadline = time.time() + 60.0
        while time.time() < deadline:
            time.sleep(0.25)
            try:
                r = requests.get(f"{BACKEND_URL}/", timeout=1.5)
                if r.status_code == 200:
                    data = r.json()
                    print(f"✅ 백엔드 준비 완료 (SAM2={data.get('sam2')}, "
                          f"Ollama={data.get('ollama')}, CSV={data.get('csv_loader')})")
                    return
            except requests.RequestException:
                pass
        print("⚠️ 백엔드 시작 시간 초과")

    start_backend()

    # ── 페이지 설정 ───────────────────────────────────────────
    page.title = "SOC 도로 포장 열응력 진단 시스템"
    page.scroll = "auto"
    page.padding = 20
    page.bgcolor = COLOR_BG

    # ════════════════════════════════════════════════════════
    # [좌측] 컨트롤 패널
    # ════════════════════════════════════════════════════════
    selected_file_label = ft.Text(
        "선택된 파일 없음", size=12, color=COLOR_TEXT_SUB,
        italic=True, selectable=True,
    )

    capture_date_field = ft.TextField(
        label="촬영일 (선택)",
        hint_text="YYYY-MM-DD",
        border_radius=8, dense=True,
    )

    # 시나리오 라디오 그룹
    scenario_radio = ft.RadioGroup(
        value="default",
        content=ft.Column([
            ft.Radio(value="default", label="운영 모드 (현장 분석)"),
            ft.Radio(value="v1",      label="V1 — 수학적 무결성 검증"),
            ft.Radio(value="v2",      label="V2 — TSRST 벤치마크"),
            ft.Radio(value="v3",      label="V3 — 극한 위험 탐지"),
        ], spacing=2)
    )

    mesh_dropdown = ft.Dropdown(
        label="격자 해상도",
        value="40",
        options=[
            ft.dropdown.Option("30",  "30 x 30 (빠름)"),
            ft.dropdown.Option("40",  "40 x 40 (기본)"),
            ft.dropdown.Option("60",  "60 x 60 (정밀)"),
            ft.dropdown.Option("80",  "80 x 80 (고정밀)"),
        ],
        border_radius=8, dense=True,
    )

    # 물성 슬라이더
    e_slider     = ft.Slider(min=1000, max=20000, value=5000, divisions=38,
                              label="E = {value} MPa", active_color=COLOR_PRIMARY)
    nu_slider    = ft.Slider(min=0.15, max=0.45, value=0.30, divisions=30,
                              label="ν = {value}", active_color=COLOR_PRIMARY)
    alpha_slider = ft.Slider(min=1.0, max=3.0, value=2.0, divisions=20,
                              label="α = {value}e-5 /K", active_color=COLOR_PRIMARY)
    st_slider    = ft.Slider(min=1.0, max=5.0, value=2.8, divisions=40,
                              label="S_t = {value} MPa", active_color=COLOR_PRIMARY)
    use_custom_props = ft.Switch(label="사용자 물성 적용", value=False, active_color=COLOR_PRIMARY)

    upload_btn  = ft.ElevatedButton("📂 이미지 업로드",
                                     bgcolor=COLOR_SECONDARY, color="white", height=42)
    analyze_btn = ft.ElevatedButton("🚀 분석 시작",
                                     bgcolor=COLOR_PRIMARY, color="white", height=46)

    status_text = ft.Text("Ready", size=12, color=COLOR_TEXT_SUB, italic=True)

    # 진행률
    STEPS = ["📥 SAM2", "🌡️ 열행렬", "📐 FEM", "⚠️ 위험도", "📊 메트릭", "🤖 LLM"]
    progress_bar = ft.ProgressBar(value=0, color=COLOR_PRIMARY,
                                   bgcolor="#E0E0E0", height=8)
    progress_pct = ft.Text("0%", size=11, weight="bold", color=COLOR_PRIMARY)
    step_indicators = [
        ft.Container(
            content=ft.Text(s, size=10, color=COLOR_TEXT_SUB),
            bgcolor="#E0E0E0", border_radius=4,
            padding=ft.Padding(6, 4, 6, 4),
        ) for s in STEPS
    ]
    progress_section = ft.Container(
        content=ft.Column([
            progress_bar,
            progress_pct,
            ft.Row(controls=step_indicators, wrap=True, spacing=4),
        ], spacing=4),
        visible=False, padding=10
    )

    def update_progress(step_idx, label=None):
        total = len(STEPS)
        pct = int(min(step_idx, total) / total * 100)
        progress_bar.value = min(step_idx, total) / total
        progress_pct.value = f"{pct}%"
        for i, ind in enumerate(step_indicators):
            if i < step_idx:
                ind.bgcolor = COLOR_SAFE; ind.content.color = "white"
            elif i == step_idx:
                ind.bgcolor = COLOR_WARNING; ind.content.color = "white"
            else:
                ind.bgcolor = "#E0E0E0"; ind.content.color = COLOR_TEXT_SUB
        page.update()

    # ════════════════════════════════════════════════════════
    # [우측] 결과 대시보드
    # ════════════════════════════════════════════════════════

    # 등급 배너
    grade_banner = ft.Container(
        content=ft.Column([
            ft.Text("🛣️ 분석을 시작하세요", size=20, weight="bold", color=COLOR_TEXT_SUB),
            ft.Text("이미지 업로드 후 [분석 시작] 버튼을 눌러주세요.",
                    size=13, color=COLOR_TEXT_SUB)
        ], spacing=4),
        bgcolor=COLOR_CARD, border_radius=10, padding=18,
        border=ft.Border.all(1, COLOR_BORDER),
    )

    # 메트릭 카드 헬퍼
    def metric_card(label, value, color, sublabel=""):
        return ft.Container(
            content=ft.Column([
                ft.Text(label, size=11, color=COLOR_TEXT_SUB, weight="bold"),
                ft.Text(value, size=20, weight="bold", color=color),
                ft.Text(sublabel, size=10, color=COLOR_TEXT_SUB),
            ], spacing=2),
            bgcolor=COLOR_CARD, border_radius=10, padding=18,
            border=ft.Border.all(1, COLOR_BORDER), expand=True,
        )

    metric_grid = ft.Row(controls=[], spacing=10)

    # 도넛/게이지 (matplotlib 이미지)
    donut_img = ft.Image(src="", width=320, height=320, visible=False)
    gauge_img = ft.Image(src="", width=420, height=260, visible=False)

    donut_section = ft.Container(
        content=ft.Column([
            ft.Text("📊 요소 분포", size=14, weight="bold", color=COLOR_PRIMARY),
            donut_img,
        ]),
        bgcolor=COLOR_CARD, border_radius=10, padding=18,
        border=ft.Border.all(1, COLOR_BORDER), expand=True,
    )
    gauge_section = ft.Container(
        content=ft.Column([
            ft.Text("🌡️ 위험도 게이지", size=14, weight="bold", color=COLOR_PRIMARY),
            gauge_img,
        ]),
        bgcolor=COLOR_CARD, border_radius=10, padding=18,
        border=ft.Border.all(1, COLOR_BORDER), expand=True,
    )

    # 정보 카드들
    scenario_card = ft.Container(
        content=ft.Column([
            ft.Text("🎯 해석 시나리오", size=14, weight="bold", color=COLOR_PRIMARY),
            ft.Text("아직 분석 안됨", size=12, color=COLOR_TEXT_SUB),
        ]),
        bgcolor=COLOR_CARD, border_radius=10, padding=18,
        border=ft.Border.all(1, COLOR_BORDER), expand=True,
    )
    cond_card = ft.Container(
        content=ft.Column([
            ft.Text("🌡️ 측정 조건 & 물성", size=14, weight="bold", color=COLOR_PRIMARY),
            ft.Text("아직 분석 안됨", size=12, color=COLOR_TEXT_SUB),
        ]),
        bgcolor=COLOR_CARD, border_radius=10, padding=18,
        border=ft.Border.all(1, COLOR_BORDER), expand=True,
    )
    uncertainty_card = ft.Container(
        content=ft.Column([
            ft.Text("📏 측정 불확실성", size=14, weight="bold", color=COLOR_PRIMARY),
            ft.Text("아직 분석 안됨", size=12, color=COLOR_TEXT_SUB),
        ]),
        bgcolor=COLOR_CARD, border_radius=10, padding=18,
        border=ft.Border.all(1, COLOR_BORDER), expand=True,
    )
    verify_card = ft.Container(
        content=ft.Column([
            ft.Text("✅ 수치해 검증", size=14, weight="bold", color=COLOR_PRIMARY),
            ft.Text("아직 분석 안됨", size=12, color=COLOR_TEXT_SUB),
        ]),
        bgcolor=COLOR_CARD, border_radius=10, padding=18,
        border=ft.Border.all(1, COLOR_BORDER), expand=True,
    )

    # 분석 이미지들
    pipeline_img     = ft.Image(src="", visible=False, width=900)
    risk_overlay_img = ft.Image(src="", visible=False, width=600)

    images_section = ft.Container(
        content=ft.Column([
            ft.Text("🔍 분석 파이프라인", size=14, weight="bold", color=COLOR_PRIMARY),
            pipeline_img,
            ft.Divider(height=8),
            ft.Text("📍 위험 위치 오버레이", size=14, weight="bold", color=COLOR_PRIMARY),
            risk_overlay_img,
        ], spacing=6),
        bgcolor=COLOR_CARD, border_radius=10, padding=18,
        border=ft.Border.all(1, COLOR_BORDER), visible=False,
    )

    # AI 진단
    ai_diag_text = ft.Text("분석 완료 후 AI 진단이 표시됩니다.",
                           size=12, color=COLOR_TEXT_SUB, selectable=True)
    ai_diag_card = ft.Container(
        content=ft.Column([
            ft.Text("🤖 AI 종합 진단", size=15, weight="bold", color=COLOR_PRIMARY),
            ft.Divider(height=4),
            ai_diag_text,
        ]),
        bgcolor="#E3F2FD", border_radius=10, padding=18,
        border=ft.Border.all(1, COLOR_SECONDARY),
    )

    # 채팅
    chat_output = ft.Column(scroll="auto", height=320, spacing=8)
    chat_input  = ft.TextField(hint_text="결과에 대해 무엇이든 물어보세요...",
                                expand=True, border_radius=20, dense=True)
    chat_send_btn = ft.ElevatedButton("전송", bgcolor=COLOR_PRIMARY,
                                        color="white", height=42)
    chat_section = ft.Container(
        content=ft.Column([
            ft.Text("💬 후속 질의응답", size=14, weight="bold", color=COLOR_PRIMARY),
            ft.Container(content=chat_output, bgcolor=COLOR_BG, border_radius=8,
                         padding=8, border=ft.Border.all(1, COLOR_BORDER), height=320),
            ft.Row([chat_input, chat_send_btn], spacing=8),
        ], spacing=8),
        bgcolor=COLOR_CARD, border_radius=10, padding=18,
        border=ft.Border.all(1, COLOR_BORDER),
    )

    # ════════════════════════════════════════════════════════
    # 결과 업데이트 함수
    # ════════════════════════════════════════════════════════
    def update_dashboard(res):
        m = res["result"]
        fem, sam2, unc = m["fem"], m["sam2"], m["uncertainty"]
        grade = m["grade"]
        props = res["props"]

        grade_color = {"Critical": COLOR_DANGER, "Warning": COLOR_WARNING,
                       "Safe": COLOR_SAFE}.get(grade, COLOR_TEXT_SUB)
        grade_bg = {"Critical": "#FFEBEE", "Warning": "#FFF8E1",
                    "Safe": "#E8F5E9"}.get(grade, COLOR_CARD)
        grade_emoji = {"Critical": "🚨", "Warning": "⚠️", "Safe": "✅"}.get(grade, "📊")

        # 등급 배너
        grade_banner.content = ft.Row([
            ft.Container(
                content=ft.Text(grade_emoji, size=40),
                bgcolor=grade_color, border_radius=50,
                padding=12, width=70, height=70,
            ),
            ft.Column([
                ft.Text(f"종합 등급: {grade}", size=24, weight="bold", color=grade_color),
                ft.Text(f"최대 응력비 SR = {fem['max_ratio']:.4f} "
                        f"(인장강도 {props['S_t']} MPa 대비 {fem['max_ratio']*100:.1f}%)",
                        size=13, color=COLOR_TEXT_MAIN),
                ft.Text(f"시나리오: {res['scenario_info']['name']}  |  "
                        f"격자: {res['mesh_resolution']}  |  "
                        f"입력: {'CSV 절대온도 ⭐' if res['input_mode']=='csv_absolute' else 'PNG 그레이스케일'}",
                        size=11, color=COLOR_TEXT_SUB),
            ], spacing=2, expand=True),
        ], spacing=16)
        grade_banner.bgcolor = grade_bg
        grade_banner.border = ft.Border.all(2, grade_color)

        # 메트릭 그리드
        metric_grid.controls = [
            metric_card("📈 최대 주응력", f"{fem['max_stress_MPa']} MPa",
                        grade_color, f"평균 {fem['mean_stress_MPa']} MPa"),
            metric_card("⚠️ Critical 요소", f"{fem['critical']}개",
                        COLOR_DANGER, f"{fem['critical_pct']}% / 전체 {fem['total']}"),
            metric_card("📍 임계 위치", f"X={fem['critical_loc_x']}%",
                        COLOR_PRIMARY, f"Y={fem['critical_loc_y']}% (좌상단 기준)"),
            metric_card("🔍 SAM2 신뢰도", f"{sam2['confidence_iou']}",
                        COLOR_SECONDARY, f"균열 {sam2['crack_pixels']}px"),
        ]

        # 이미지 갱신 (도넛/게이지/분석이미지 모두)
        t = int(time.time())
        donut_img.src = f"{BACKEND_URL_PUBLIC}/static/donut.png?t={t}"
        donut_img.visible = True
        gauge_img.src = f"{BACKEND_URL_PUBLIC}/static/gauge.png?t={t}"
        gauge_img.visible = True

        # 시나리오 카드
        scenario_card.content = ft.Column([
            ft.Text("🎯 해석 시나리오", size=14, weight="bold", color=COLOR_PRIMARY),
            ft.Text(res['scenario_info']['name'], size=13, weight="bold"),
            ft.Text(res['scenario_info']['desc'], size=11, color=COLOR_TEXT_SUB),
            ft.Divider(height=4),
            ft.Text(f"📐 격자: {res['mesh_resolution']}", size=11),
            ft.Text(f"📥 입력: {res['input_mode']}", size=11),
        ], spacing=2)

        # 측정 조건 카드
        season = "혹한기" if props["base_temp"] < 0 else "환절기" if props["base_temp"] < 15 else "하절기"
        cond_card.content = ft.Column([
            ft.Text("🌡️ 측정 조건 & 물성", size=14, weight="bold", color=COLOR_PRIMARY),
            ft.Text(f"📅 촬영일: {res.get('capture_date','?')} "
                    f"({props['month']}월, {season})", size=11),
            ft.Text(f"🌡️ 월평균 최저: {props['base_temp']}°C", size=11),
            ft.Divider(height=4),
            ft.Text("적용 물성:", size=11, weight="bold", color=COLOR_TEXT_SUB),
            ft.Text(f"  • E = {props['E']} MPa", size=11),
            ft.Text(f"  • ν = {props['nu']}", size=11),
            ft.Text(f"  • α = {props['alpha']:.2e} /K", size=11),
            ft.Text(f"  • S_t = {props['S_t']} MPa", size=11),
        ], spacing=2)

        # 불확실성 카드
        uncertainty_card.content = ft.Column([
            ft.Text("📏 측정 불확실성 (FLIR ±3°C)", size=14, weight="bold", color=COLOR_PRIMARY),
            ft.Text(f"센서 오차: ±{unc['delta_T_C']}°C", size=11),
            ft.Text(f"응력 변동: ±{unc['delta_sigma_MPa']:.3f} MPa", size=11),
            ft.Divider(height=4),
            ft.Text("응력 신뢰구간:", size=11, weight="bold", color=COLOR_TEXT_SUB),
            ft.Text(f"  {unc['stress_range_MPa'][0]} ~ {unc['stress_range_MPa'][1]} MPa", size=11),
            ft.Text("응력비 신뢰구간:", size=11, weight="bold", color=COLOR_TEXT_SUB),
            ft.Text(f"  {unc['SR_range'][0]} ~ {unc['SR_range'][1]}", size=11),
        ], spacing=2)

        # 검증 카드
        ver = res["verification"]
        v1_err = ver.get("V1_error_pct", 0)
        v1_pass = v1_err < 15
        verify_card.content = ft.Column([
            ft.Text("✅ 수치해 검증", size=14, weight="bold", color=COLOR_PRIMARY),
            ft.Text(f"V1 이론해 오차: {v1_err:.2f}% "
                    f"[{'PASS ✅' if v1_pass else 'REVIEW ⚠️'}]",
                    size=12, weight="bold",
                    color=COLOR_SAFE if v1_pass else COLOR_WARNING),
            ft.Text(f"수식: σ = E·α·ΔT/(1-ν), ΔT=15°C", size=10, color=COLOR_TEXT_SUB),
            ft.Divider(height=4),
            ft.Text("V2 TSRST 비교:", size=11, weight="bold"),
            ft.Text(f"  시뮬 {ver.get('V2_sim_stress_MPa',0):.2f} MPa "
                    f"vs 문헌 {ver.get('V2_tsrst_literature_MPa',0)} MPa", size=11),
            ft.Text(f"  오차: {ver.get('V2_tsrst_error_pct', 0):.2f}%", size=11),
        ], spacing=2)

        # 분석 이미지
        pipeline_img.src     = f"{BACKEND_URL_PUBLIC}/static/pipeline_result.png?t={t}"
        pipeline_img.visible = True
        risk_overlay_img.src = f"{BACKEND_URL_PUBLIC}/static/risk_overlay.png?t={t}"
        risk_overlay_img.visible = True
        images_section.visible = True

    # ════════════════════════════════════════════════════════
    # 핸들러
    # ════════════════════════════════════════════════════════
    def _send_to_backend(local_path, name):
        size = os.path.getsize(local_path) if os.path.exists(local_path) else -1
        print(f"[Upload] ⏫ 백엔드 전송 시작: name={name} size={size}B src={local_path}", flush=True)
        try:
            with open(local_path, "rb") as f:
                r = requests.post(f"{BACKEND_URL}/upload",
                                   files={"file": (name, f)}, timeout=120)
            if r.status_code == 200:
                data = r.json()
                ext = data.get("ext", "")
                saved = data.get("path", "?")
                hint = " (CSV 절대온도 ⭐)" if ext == ".csv" else ""
                status_text.value = f"✅ 업로드 성공!{hint}"
                print(f"[Upload] ✅ 완료: name={name} ext={ext} saved={saved}", flush=True)
            else:
                status_text.value = f"❌ 업로드 실패 ({r.status_code})"
                print(f"[Upload] ❌ 백엔드 HTTP {r.status_code}: {r.text[:200]}", flush=True)
        except Exception as ex:
            status_text.value = f"❌ 업로드 오류: {ex}"
            print(f"[Upload] ❌ 예외: {ex}", flush=True)
        page.update()

    def on_upload_progress(e):
        if e.error:
            status_text.value = f"❌ 업로드 실패: {e.error}"
            page.update(); return
        if e.progress is not None and e.progress < 1.0:
            status_text.value = f"⬆️ {e.file_name}: {int(e.progress*100)}%"
            page.update()

    picker = ft.FilePicker(on_upload=on_upload_progress)
    page.services.append(picker)

    async def on_upload_click(e):
        files = await picker.pick_files(
            allow_multiple=False,
            allowed_extensions=["png", "jpg", "jpeg", "csv", "tif", "tiff", "bmp"],
        )
        if not files:
            status_text.value = "취소됨"
            page.update()
            print("[Upload] 파일 선택 취소됨", flush=True)
            return
        f = files[0]
        selected_file_label.value = f"📄 {f.name}"
        page.update()
        if f.path:
            print(f"[Upload] 📥 선택: name={f.name} mode=desktop path={f.path}", flush=True)
            status_text.value = f"📤 업로드 중: {f.name}..."
            page.update()
            _send_to_backend(f.path, f.name)
        else:
            print(f"[Upload] 📥 선택: name={f.name} mode=web (브라우저→서버 전송 필요)", flush=True)
            status_text.value = f"⬆️ 브라우저 → 서버 전송 중: {f.name}..."
            page.update()
            await picker.upload([
                ft.FilePickerUploadFile(
                    name=f.name,
                    upload_url=page.get_upload_url(f.name, 600),
                )
            ])
            server_path = os.path.join(UPLOAD_DIR, f.name)
            print(f"[Upload] 💾 브라우저→서버 전송 완료: {server_path}", flush=True)
            status_text.value = f"📤 백엔드 전달 중: {f.name}..."
            page.update()
            _send_to_backend(server_path, f.name)

    def run_analysis_thread():
        try:
            progress_section.visible = True
            images_section.visible = False
            ai_diag_text.value = "분석 중... 잠시만 기다려주세요."
            update_progress(0, "백엔드 호출 중...")
            status_text.value = "🔍 SAM2 + FEM 해석 중..."
            page.update()

            params = {
                "scenario": scenario_radio.value or "default",
                "mesh_size": mesh_dropdown.value,
            }
            cd = (capture_date_field.value or "").strip()
            if cd:
                params["capture_date"] = cd
            if use_custom_props.value:
                params["e_override"] = e_slider.value
                params["nu_override"] = nu_slider.value
                params["alpha_override"] = alpha_slider.value * 1e-5
                params["st_override"] = st_slider.value

            r = requests.get(f"{BACKEND_URL}/analyze", params=params, timeout=300)
            res = r.json()

            if "error" in res:
                status_text.value = f"❌ 오류: {res['error']}"
                ai_diag_text.value = f"분석 실패: {res['error']}"
                if "detail" in res:
                    print("상세:\n", res["detail"])
                page.update(); return

            update_progress(4, "메트릭 계산 완료")
            update_dashboard(res)
            status_text.value = "✅ FEM 완료, AI 진단 생성 중..."
            ai_diag_text.value = "🤖 AI 진단 생성 중... (1~2분 소요됩니다)"
            page.update()

            update_progress(5, "🤖 LLM 진단 중...")
            try:
                rd = requests.get(f"{BACKEND_URL}/diagnose", timeout=600)
                drez = rd.json()
                ai_diag_text.value = drez.get("diagnosis") or drez.get("error", "진단 결과 없음")
            except requests.exceptions.Timeout:
                ai_diag_text.value = "⚠️ LLM 응답 타임아웃"
            except Exception as ex:
                ai_diag_text.value = f"⚠️ 진단 오류: {ex}"

            update_progress(6)
            status_text.value = "✅ 분석 완료!"

        except requests.exceptions.Timeout:
            status_text.value = "❌ 타임아웃"
            ai_diag_text.value = "⚠️ 분석이 너무 오래 걸려 중단되었습니다."
        except requests.exceptions.ConnectionError:
            status_text.value = "❌ 백엔드 연결 실패"
            ai_diag_text.value = "백엔드(main.py)가 실행 중인지 확인하세요."
        except Exception as ex:
            status_text.value = f"❌ 오류: {ex}"
            ai_diag_text.value = f"⚠️ {ex}"

        page.update()

    def analyze_click(e):
        threading.Thread(target=run_analysis_thread, daemon=True).start()

    def send_chat(e):
        msg = chat_input.value.strip()
        if not msg: return
        chat_output.controls.append(
            ft.Container(
                content=ft.Text(f"🧑 {msg}", size=12, weight="bold", selectable=True),
                bgcolor="#E8EAF6", border_radius=8, padding=10,
            )
        )
        chat_input.value = ""
        page.update()
        try:
            r = requests.post(f"{BACKEND_URL}/chat",
                              json={"message": msg}, timeout=180)
            reply = r.json().get("reply", "응답 없음")
        except requests.exceptions.Timeout:
            reply = "⚠️ 응답 시간 초과"
        except Exception as ex:
            reply = f"⚠️ 오류: {ex}"
        chat_output.controls.append(
            ft.Container(
                content=ft.Text(f"🤖 {reply}", size=12, selectable=True),
                bgcolor="#F1F8E9", border_radius=8, padding=10,
            )
        )
        page.update()

    upload_btn.on_click = on_upload_click
    analyze_btn.on_click = analyze_click
    chat_send_btn.on_click = send_chat
    chat_input.on_submit = send_chat

    # ════════════════════════════════════════════════════════
    # 좌측 패널 조립
    # ════════════════════════════════════════════════════════
    left_panel = ft.Container(
        content=ft.Column([
            ft.Text("🛣️ SOC", size=24, weight="bold", color=COLOR_PRIMARY),
            ft.Text("도로 포장 열응력 진단 v2.0", size=12, color=COLOR_TEXT_SUB),
            ft.Divider(height=8),

            ft.Text("📥 1. 이미지 입력", size=13, weight="bold", color=COLOR_PRIMARY),
            upload_btn,
            selected_file_label,
            capture_date_field,

            ft.Divider(height=8),
            ft.Text("⚙️ 2. 해석 시나리오", size=13, weight="bold", color=COLOR_PRIMARY),
            scenario_radio,

            ft.Divider(height=8),
            ft.Text("📐 3. 격자 해상도", size=13, weight="bold", color=COLOR_PRIMARY),
            mesh_dropdown,

            ft.Divider(height=8),
            ft.Text("⚙️ 4. 물성치 (선택)", size=13, weight="bold", color=COLOR_PRIMARY),
            use_custom_props,
            ft.Text("탄성계수 E (MPa)", size=11, color=COLOR_TEXT_SUB),
            e_slider,
            ft.Text("포아송비 ν", size=11, color=COLOR_TEXT_SUB),
            nu_slider,
            ft.Text("열팽창 α (×10⁻⁵ /K)", size=11, color=COLOR_TEXT_SUB),
            alpha_slider,
            ft.Text("인장강도 S_t (MPa)", size=11, color=COLOR_TEXT_SUB),
            st_slider,

            ft.Divider(height=12),
            analyze_btn,
            status_text,
            progress_section,
        ], spacing=6, scroll="auto"),
        bgcolor=COLOR_CARD, border_radius=12, padding=18,
        border=ft.Border.all(1, COLOR_BORDER),
        width=380,
    )

    # ════════════════════════════════════════════════════════
    # 우측 패널 조립
    # ════════════════════════════════════════════════════════
    right_panel = ft.Column([
        grade_banner,
        metric_grid,
        ft.Row([gauge_section, donut_section], spacing=12),
        ft.Row([scenario_card, cond_card], spacing=12),
        ft.Row([uncertainty_card, verify_card], spacing=12),
        ai_diag_card,
        images_section,
        chat_section,
    ], spacing=12, expand=True, scroll="auto")

    # ════════════════════════════════════════════════════════
    # 최종 레이아웃
    # ════════════════════════════════════════════════════════
    page.add(
        ft.Row([
            left_panel,
            ft.Container(content=right_panel, expand=True),
        ], spacing=16, expand=True)
    )

if os.environ.get("FLET_WEB", "").lower() in ("1", "true", "yes"):
    ft.run(
        main,
        view=ft.AppView.WEB_BROWSER,
        host=os.environ.get("FLET_HOST", "0.0.0.0"),
        port=int(os.environ.get("FLET_PORT", "8550")),
        upload_dir=UPLOAD_DIR,
    )
else:
    ft.run(main, upload_dir=UPLOAD_DIR)
