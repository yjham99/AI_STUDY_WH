import os
import subprocess
import requests
import json

LOG_FILE = "execution_log.txt"
OLLAMA_MODEL = "gemma2:9b"

def get_latest_log():
    try:
        if not os.path.exists(LOG_FILE):
            return "로그 파일이 존재하지 않습니다."
        
        with open(LOG_FILE, "r") as f:
            content = f.read()
            # 마지막 실행 세션만 추출 (--- 실행 시각 기준)
            sessions = content.split("--- 실행 시각")
            if len(sessions) > 1:
                return sessions[-1]
            return content
    except Exception as e:
        return f"로그 읽기 실패: {str(e)}"

def summarize_with_ollama(text):
    print(f"🤖 Ollama ({OLLAMA_MODEL})를 사용하여 로그 요약 중...")
    
    url = "http://localhost:11434/api/generate"
    prompt = f"""
    아래는 AI 투자 시스템의 실행 로그입니다. 
    이 중에서 개발자가 알아야 할 핵심 에러나 결과만 500자 이내로 요약해 주세요.
    영어로 된 에러 메시지가 있다면 핵심 의미를 한국어로 설명해 주세요.

    로그 내용:
    {text}
    """
    
    data = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            return response.json().get("response", "요약을 생성하지 못했습니다.")
        else:
            return f"Ollama 응답 오류: {response.status_code}"
    except Exception as e:
        return f"Ollama 연결 실패: {str(e)}"

if __name__ == "__main__":
    log_text = get_latest_log()
    summary = summarize_with_ollama(log_text)
    
    print("\n" + "="*50)
    print("📝 로그 요약 결과 (Jules에게 전달용)")
    print("="*50)
    print(summary)
    print("="*50)
