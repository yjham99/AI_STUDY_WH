"""
키움증권 OAuth 2.0 토큰 인증 관리 모듈
- 용도: 접근 토큰 발급(au10001), 폐기(au10002), 만료 감지(8005 대응)
- TODO: 내일 실제 앱 키/시크릿 키 발급 후 .env에 추가하여 연동
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

# 환경 설정: .env의 KIWOOM_ENV 값에 따라 실전/모의 분기
KIWOOM_ENV = os.getenv("KIWOOM_ENV", "mock")
BASE_URL = "https://mockapi.kiwoom.com" if KIWOOM_ENV == "mock" else "https://api.kiwoom.com"

APP_KEY    = os.getenv("KIWOOM_APP_KEY", "")
APP_SECRET = os.getenv("KIWOOM_APP_SECRET", "")


class KiwoomAuth:
    """키움증권 OAuth 2.0 인증 토큰 생명주기 관리 클래스"""

    def __init__(self):
        self.token = None

    def get_token(self) -> str:
        """
        [au10001] 접근 토큰 발급
        - App Key + Secret Key → Bearer Token 획득
        - 반환된 토큰 내부에 저장 후 반환
        """
        # TODO: 실제 API 호출 구현 (내일 진행)
        url = f"{BASE_URL}/oauth2/token"
        payload = {
            "grant_type": "client_credentials",
            "appkey": APP_KEY,
            "secretkey": APP_SECRET,
        }
        headers = {
            "Content-Type": "application/json;charset=UTF-8",
            "api-id": "au10001"
        }
        response = requests.post(url, json=payload, headers=headers)
        data = response.json()
        self.token = data.get("token", "")
        print(f"✅ 토큰 발급 완료 (env: {KIWOOM_ENV})")
        return self.token

    def revoke_token(self):
        """
        [au10002] 접근 토큰 폐기
        - 세션 종료 시 반드시 호출하여 리플레이 공격 방지
        """
        # TODO: 실제 API 호출 구현 (내일 진행)
        if not self.token:
            print("⚠️ 폐기할 토큰이 없습니다.")
            return
        url = f"{BASE_URL}/oauth2/revoke"
        payload = {
            "appkey": APP_KEY,
            "secretkey": APP_SECRET,
            "token": self.token
        }
        headers = {
            "Content-Type": "application/json;charset=UTF-8",
            "api-id": "au10002",
            "authorization": f"Bearer {self.token}"
        }
        response = requests.post(url, json=payload, headers=headers)
        print(f"✅ 토큰 폐기 완료: {response.json()}")
        self.token = None

    def get_auth_headers(self, api_id: str) -> dict:
        """모든 REST API 요청에 사용할 공통 헤더 반환"""
        return {
            "Content-Type": "application/json;charset=UTF-8",
            "authorization": f"Bearer {self.token}",
            "api-id": api_id,
        }


if __name__ == "__main__":
    auth = KiwoomAuth()
    token = auth.get_token()
    print(f"발급 토큰: {token[:20]}...")
