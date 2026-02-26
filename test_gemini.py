import os
from dotenv import load_dotenv
import google.generativeai as genai

# .env 파일 로드
load_dotenv()

# API 키 설정 (TIER1 사용)
api_key = os.getenv("GEMINI_API_KEY_TIER1")
if not api_key:
    print("Error: GEMINI_API_KEY_FREE1 not found in .env")
else:
    genai.configure(api_key=api_key)
    
    try:
        # 모델 설정
        model = genai.GenerativeModel("gemini-2.0-flash")
        
        # 테스트 메시지 전송
        print("Gemini API 연결 테스트 중...")
        response = model.generate_content("안녕하세요! 연결 테스트입니다. 짧게 인사해 주세요.")
        
        print("\n[Gemini 응답]")
        print(response.text)
        print("\n✅ API 연결 성공!")
        
    except Exception as e:
        print(f"\n❌ API 연결 실패: {str(e)}")
