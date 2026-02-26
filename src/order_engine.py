"""
주문 집행 엔진 (Order Execution Engine)
- kt10000: 매수 주문
- kt10001: 매도 주문
- kt10002: 주문 정정
- kt10003: 주문 취소 (cncl_qty=0 → 잔량 전체 취소)
- ka10075: 미체결 동기화 (자가치유 로직)
"""
import os
import requests
from src.kiwoom_auth import KiwoomAuth

KIWOOM_ENV = os.getenv("KIWOOM_ENV", "mock")
BASE_URL = "https://mockapi.kiwoom.com" if KIWOOM_ENV == "mock" else "https://api.kiwoom.com"
# 모의투자는 KRX만 지원, 실전은 SOR(최선주문집행) 기본
DEFAULT_EXCHANGE = "KRX" if KIWOOM_ENV == "mock" else "SOR"


class OrderEngine:
    """AI 신호 → 주문 실행 변환 엔진"""

    def __init__(self, auth: KiwoomAuth):
        self.auth = auth
        self.pending_orders = {}  # {ord_no: {stk_cd, qty, price}}

    def buy(self, stk_cd: str, qty: int, price: int = 0, order_type: str = "00") -> str:
        """
        [kt10000] 매수 주문
        - order_type: 00=지정가, 03=시장가(price=0), 20=FOK
        - 반환: ord_no (주문번호, 취소/정정의 외래키)
        """
        # TODO: 실제 API 호출 구현
        url = f"{BASE_URL}/api/dostk/ordr"
        headers = self.auth.get_auth_headers("kt10000")
        payload = {
            "dmst_stex_tp": DEFAULT_EXCHANGE,
            "stk_cd": stk_cd,
            "ord_qty": str(qty),
            "ord_uv": str(price) if order_type != "03" else "",
            "trde_tp": order_type,
        }
        response = requests.post(url, json=payload, headers=headers)
        data = response.json()
        ord_no = data.get("ord_no", "")
        self.pending_orders[ord_no] = {"stk_cd": stk_cd, "qty": qty, "price": price}
        print(f"✅ 매수 주문 접수: {stk_cd} {qty}주 @ {price}원 (주문번호: {ord_no})")
        return ord_no

    def sell(self, stk_cd: str, qty: int, price: int = 0, order_type: str = "00") -> str:
        """
        [kt10001] 매도 주문
        """
        # TODO: 실제 API 호출 구현
        url = f"{BASE_URL}/api/dostk/ordr"
        headers = self.auth.get_auth_headers("kt10001")
        payload = {
            "dmst_stex_tp": DEFAULT_EXCHANGE,
            "stk_cd": stk_cd,
            "ord_qty": str(qty),
            "ord_uv": str(price) if order_type != "03" else "",
            "trde_tp": order_type,
        }
        response = requests.post(url, json=payload, headers=headers)
        data = response.json()
        ord_no = data.get("ord_no", "")
        print(f"✅ 매도 주문 접수: {stk_cd} {qty}주 @ {price}원 (주문번호: {ord_no})")
        return ord_no

    def cancel(self, orig_ord_no: str, stk_cd: str, qty: int = 0):
        """
        [kt10003] 주문 취소
        - qty=0: 해당 주문번호의 미체결 잔량 전체 일괄 취소 (매크로 기능)
        """
        # TODO: 실제 API 호출 구현
        url = f"{BASE_URL}/api/dostk/ordr"
        headers = self.auth.get_auth_headers("kt10003")
        payload = {
            "orig_ord_no": orig_ord_no,
            "stk_cd": stk_cd,
            "cncl_qty": str(qty),  # 0이면 잔량 전체 취소
        }
        response = requests.post(url, json=payload, headers=headers)
        print(f"✅ 주문 취소 완료: {orig_ord_no}")
        if orig_ord_no in self.pending_orders:
            del self.pending_orders[orig_ord_no]

    def sync_pending_orders(self):
        """
        [ka10075] 미체결 동기화 (자가치유 로직)
        - 주기적 폴링으로 내부 장부와 서버 상태 일치화
        - 'Orphaned Order' 감지: 주문단가 vs 현재가 임계치 초과 시 자동 취소
        """
        # TODO: 실제 API 호출 및 자가치유 로직 구현
        url = f"{BASE_URL}/api/dostk/oso"
        headers = self.auth.get_auth_headers("ka10075")
        response = requests.post(url, json={}, headers=headers)
        data = response.json()
        print(f"📋 미체결 주문 동기화 완료: {len(data.get('oso', []))}건")
        return data.get("oso", [])
