import os
import time
from dotenv import load_dotenv
from google import genai

# .env 파일 로드
load_dotenv()

# 진단 목록에서 확인된 사용 가능 모델 (우선순위 순)
MODELS_TO_TRY = ["gemini-2.0-flash", "gemini-2.5-flash", "models/gemini-2.0-flash"]

def run_test(api_key_name):
    api_key = os.getenv(api_key_name)
    if not api_key:
        print(f"⚠️ {api_key_name}를 찾을 수 없습니다.")
        return False, None

    client = genai.Client(api_key=api_key)
    
    for model_name in MODELS_TO_TRY:
        try:
            print(f"[{api_key_name}] {model_name} 모델로 시도 중...")
            response = client.models.generate_content(
                model=model_name,
                contents="반갑습니다! 연결 성공인가요? 짧게 답장 부탁드려요."
            )
            return True, f"({model_name}) {response.text}"
        except Exception as e:
            err_msg = str(e)
            if "404" in err_msg:
                print(f"   ㄴ ❌ {model_name} 실패 (404). 다음 모델 확인...")
                continue
            elif "429" in err_msg:
                return False, f"할당량 초과 (429)"
            else:
                return False, err_msg
                
    return False, "연결 가능한 모델을 찾지 못했습니다."

# --- 메인 실행 로직 ---
print("🚀 Gemini API 최적화 전략 (Free -> Tier1)")

# 1. Free 키 테스트
success, message = run_test("GEMINI_API_KEY_FREE1")

if success:
    print(f"\n✅ Free API 연결 성공!\n[응답]: {message}")
else:
    print(f"\n❌ Free API 실패: {message}")
    print("🔄 Tier1 (유급) 키로 전환합니다...")
    
    # 2. Tier1 키 테스트
    success_tier, message_tier = run_test("GEMINI_API_KEY_TIER1")
    
    if success_tier:
        print(f"\n✅ Tier1 API 연결 성공!\n[응답]: {message_tier}")
    else:
        print(f"\n❌ 최종 실패: {message_tier}")
