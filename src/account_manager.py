"""
계좌/예수금/잔고 관리 모듈
- ka00001: 계좌번호 동적 조회
- kt00001: 예수금/유동성 조회 (매수 가능 금액, 미수금 감지)
- kt00005: 체결잔고/포트폴리오 현황
"""
import requests
from src.kiwoom_auth import KiwoomAuth

class AccountManager:
    """AI 상태 공간(State Space) 구성용 계좌 관리 클래스"""

    def __init__(self, auth: KiwoomAuth):
        self.auth = auth
        self.base_url = "https://mockapi.kiwoom.com"  # auth 모듈에서 상속 예정
        self.account_no = None

    def get_account_number(self) -> str:
        """
        [ka00001] 계좌번호 동적 조회
        - 하드코딩 금지! 토큰에 바인딩된 계좌번호를 동적으로 식별
        """
        # TODO: 실제 API 호출 구현
        url = f"{self.base_url}/api/dostk/acnt"
        headers = self.auth.get_auth_headers("ka00001")
        response = requests.post(url, json={}, headers=headers)
        data = response.json()
        self.account_no = data.get("acctNo", "")
        print(f"✅ 계좌번호 식별: {self.account_no}")
        return self.account_no

    def get_balance(self) -> dict:
        """
        [kt00001] 예수금 상세 조회
        - AI 리스크 관리 핵심 데이터
        - ch_uncla(미수금) > 0 이면 즉시 청산 모드 진입
        - ord_alow_amt: 포지션 사이징 상한선
        """
        # TODO: 실제 API 호출 구현
        url = f"{self.base_url}/api/dostk/acnt"
        headers = self.auth.get_auth_headers("kt00001")
        payload = {"qry_tp": "2"}  # 일반 조회
        response = requests.post(url, json=payload, headers=headers)
        data = response.json()

        balance = {
            "available_cash": data.get("entr", 0),
            "order_available": data.get("ord_alow_amt", 0),
            "unpaid_cash": data.get("ch_uncla", 0),
        }

        # 리스크 체크
        if int(balance["unpaid_cash"]) > 0:
            print("🚨 미수금 감지! 즉시 청산 모드 진입 필요.")

        return balance

    def get_portfolio(self) -> dict:
        """
        [kt00005] 체결잔고(보유 포트폴리오) 조회
        - tot_pl_rt: 총 수익률 (일일 손실 제한 체크)
        - stk_cntr_remn: 종목별 보유수량, 평균단가, 평가손익
        """
        # TODO: 실제 API 호출 구현
        url = f"{self.base_url}/api/dostk/acnt"
        headers = self.auth.get_auth_headers("kt00005")
        payload = {"dmst_stex_tp": "KRX"}
        response = requests.post(url, json=payload, headers=headers)
        return response.json()


if __name__ == "__main__":
    auth = KiwoomAuth()
    auth.get_token()
    mgr = AccountManager(auth)
    mgr.get_account_number()
    balance = mgr.get_balance()
    print("잔고:", balance)
