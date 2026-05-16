"""
Ollama 연결 문제 진단 스크립트
실행: python diagnose_ollama.py
"""

import sys
import socket
import subprocess
import requests
import json
import time
import os

SEPARATOR = "=" * 60

def section(title):
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)

def ok(msg):   print(f"  ✅ {msg}")
def fail(msg): print(f"  ❌ {msg}")
def info(msg): print(f"  ℹ️  {msg}")
def warn(msg): print(f"  ⚠️  {msg}")

# ────────────────────────────────────────────────
# [1] 포트 11434 열려있는지 확인
# ────────────────────────────────────────────────
section("1단계: 포트 11434 오픈 여부 확인")
try:
    sock = socket.create_connection(("127.0.0.1", 11434), timeout=3)
    sock.close()
    ok("포트 11434 열려 있음 → Ollama 서버 실행 중")
    port_open = True
except Exception as e:
    fail(f"포트 11434 닫혀 있음 → Ollama 서버가 실행되지 않았습니다!")
    info("해결: 별도 터미널에서 'ollama serve' 실행 후 다시 시도하세요.")
    port_open = False

# ────────────────────────────────────────────────
# [2] Ollama HTTP API 응답 확인
# ────────────────────────────────────────────────
section("2단계: Ollama HTTP API 응답 확인")
api_ok = False
if port_open:
    for url in ["http://localhost:11434/api/tags", "http://127.0.0.1:11434/api/tags"]:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                ok(f"API 응답 정상: {url}")
                models = r.json().get("models", [])
                info(f"설치된 모델 수: {len(models)}개")
                for m in models:
                    info(f"  - {m['name']} ({round(m.get('size',0)/1e9, 2)} GB)")
                api_ok = True
                break
            else:
                fail(f"API 응답 오류 (status={r.status_code}): {url}")
        except requests.exceptions.ConnectionError:
            fail(f"연결 거부됨: {url}")
        except Exception as e:
            fail(f"예외 발생: {e}")
    if not api_ok:
        info("해결: 방화벽이나 localhost 바인딩 문제일 수 있습니다.")
else:
    warn("포트가 닫혀있어 API 확인 생략")

# ────────────────────────────────────────────────
# [3] exaone3.5:7.8b 모델 존재 여부 확인
# ────────────────────────────────────────────────
section("3단계: exaone3.5:7.8b 모델 설치 확인")
model_ok = False
if api_ok:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        if "exaone3.5:7.8b" in models:
            ok("exaone3.5:7.8b 모델 설치 확인됨")
            model_ok = True
        else:
            fail(f"exaone3.5:7.8b 없음. 설치된 모델: {models}")
            info("해결: 'ollama pull exaone3.5:7.8b' 실행")
    except Exception as e:
        fail(f"모델 목록 조회 실패: {e}")
else:
    warn("API 응답 없어 모델 확인 생략")

# ────────────────────────────────────────────────
# [4] 실제 채팅 API 호출 테스트 (짧은 프롬프트)
# ────────────────────────────────────────────────
section("4단계: 실제 /api/chat 호출 테스트")
if model_ok:
    print("  ⏳ LLM 호출 중... (최대 60초 대기)")
    try:
        start = time.time()
        r = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "exaone3.5:7.8b",
                "messages": [{"role": "user", "content": "1+1은?"}],
                "stream": False
            },
            timeout=60
        )
        elapsed = round(time.time() - start, 2)
        if r.status_code == 200:
            reply = r.json()["message"]["content"]
            ok(f"LLM 응답 성공! ({elapsed}초)")
            info(f"응답: {reply[:100]}")
        else:
            fail(f"LLM 응답 실패 (status={r.status_code})")
            info(f"응답 내용: {r.text[:200]}")
    except requests.exceptions.Timeout:
        fail("타임아웃 (60초 초과) → GPU/CPU 성능 부족 또는 모델 로딩 중")
        info("해결: 첫 실행 시 모델 로딩에 시간이 걸립니다. timeout을 180초로 늘려보세요.")
    except Exception as e:
        fail(f"예외 발생: {e}")
else:
    warn("모델 없어 채팅 테스트 생략")

# ────────────────────────────────────────────────
# [5] 방화벽 / 프로세스 확인 (Windows)
# ────────────────────────────────────────────────
section("5단계: Ollama 프로세스 및 방화벽 확인 (Windows)")
if os.name == 'nt':
    # 프로세스 확인
    try:
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq ollama.exe"],
            capture_output=True, text=True
        )
        if "ollama.exe" in result.stdout:
            ok("ollama.exe 프로세스 실행 중")
        else:
            fail("ollama.exe 프로세스 없음 → Ollama가 실행되지 않았습니다")
            info("해결: 'ollama serve' 명령어를 별도 터미널에서 실행하세요")
    except Exception as e:
        warn(f"프로세스 확인 실패: {e}")

    # 포트 사용 프로세스 확인
    try:
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True, text=True
        )
        lines_11434 = [l for l in result.stdout.splitlines() if ":11434" in l]
        if lines_11434:
            ok("11434 포트 사용 중인 프로세스:")
            for l in lines_11434[:3]:
                info(f"  {l.strip()}")
        else:
            fail("11434 포트를 사용 중인 프로세스 없음")
    except Exception as e:
        warn(f"netstat 확인 실패: {e}")
else:
    warn("Windows 환경이 아니어서 프로세스 확인 생략")

# ────────────────────────────────────────────────
# [6] OLLAMA_HOST 환경변수 확인
# ────────────────────────────────────────────────
section("6단계: 환경변수 확인")
ollama_host = os.environ.get("OLLAMA_HOST", "설정 안됨")
info(f"OLLAMA_HOST = {ollama_host}")
if ollama_host != "설정 안됨" and "11434" not in ollama_host:
    warn("OLLAMA_HOST가 다른 포트로 설정되어 있습니다!")
    info(f"main.py의 URL을 http://{ollama_host}/api/chat 으로 변경해보세요.")
else:
    ok("OLLAMA_HOST 환경변수 정상 (기본값 사용)")

# ────────────────────────────────────────────────
# [최종] 요약
# ────────────────────────────────────────────────
section("🔍 진단 요약 및 권장 조치")
if not port_open:
    print("""
  → 포트 11434이 닫혀 있습니다.
  
  [조치 1] 새 터미널(cmd)을 열고 아래 명령어 실행:
           ollama serve
  
  [조치 2] 위 터미널을 닫지 말고, 다른 터미널에서 app.py 실행
""")
elif not api_ok:
    print("""
  → 포트는 열려있지만 API 응답이 없습니다.
  
  [조치 1] Windows Defender / 방화벽에서 ollama.exe 허용
  [조치 2] 관리자 권한으로 cmd 실행 후 'ollama serve' 재시작
  [조치 3] 127.0.0.1 대신 0.0.0.0 으로 바인딩:
           set OLLAMA_HOST=0.0.0.0:11434
           ollama serve
""")
elif not model_ok:
    print("""
  → 모델이 설치되지 않았습니다.
  
  [조치] ollama pull exaone3.5:7.8b
""")
else:
    print("""
  → 모든 항목 정상! main.py의 timeout 값을 120~180으로 올려보세요.
  
  call_ollama(prompt, timeout=180)
""")

print(f"\n{SEPARATOR}\n")
