"""
WebSocket 기반 실시간 시세 스트림 모듈
- 실시간 체결/호가/주문체결 알림 수신
- asyncio + websockets 기반 비동기 처리
"""
import asyncio
import json
import os
import websockets
from src.kiwoom_auth import KiwoomAuth

KIWOOM_ENV = os.getenv("KIWOOM_ENV", "mock")
WS_URL = (
    "wss://mockapi.kiwoom.com:10000/api/dostk/websocket"
    if KIWOOM_ENV == "mock"
    else "wss://api.kiwoom.com:10000/api/dostk/websocket"
)


class RealtimeStream:
    """
    WebSocket 실시간 데이터 수신 클래스
    - ID 00: 주문체결 알림 (이벤트 드리븐 트리거)
    - ID 0B: 주식체결 틱 데이터 (체결강도, VWAP)
    - ID 0D: 호가잔량 Level 2 (시장 미시구조)
    """

    def __init__(self, auth: KiwoomAuth):
        self.auth = auth
        self.subscribed_stocks = []

    def build_subscribe_payload(self, stock_codes: list, stream_types: list) -> dict:
        """구독 요청 페이로드 생성"""
        return {
            "trnm": "REG",
            "refresh": "1",  # 기존 목록에 병합
            "data": [
                {"item": stock_codes, "type": t} for t in stream_types
            ]
        }

    async def listen(self, stock_codes: list):
        """
        WebSocket 연결 및 실시간 데이터 수신 루프
        - 0B: 틱 체결 데이터 (체결강도 FID 228, 체결수량 부호로 수급 방향 판별)
        - 0D: 호가창(Order Book) 1~10호가 매수/매도 잔량
        """
        headers = {
            "authorization": f"Bearer {self.auth.token}",
            "api-id": "ws00001"
        }
        async with websockets.connect(WS_URL, extra_headers=headers) as ws:
            # 구독 등록 (체결 + 호가)
            subscribe_msg = self.build_subscribe_payload(
                stock_codes=stock_codes,
                stream_types=["0B", "0D"]  # 틱 + 호가잔량
            )
            await ws.send(json.dumps(subscribe_msg))
            print(f"📡 실시간 스트림 구독 시작: {stock_codes}")

            async for message in ws:
                data = json.loads(message)
                await self.handle_message(data)

    async def handle_message(self, data: dict):
        """
        수신 메시지 파싱 및 처리
        - FID 15(체결수량) 부호: 양수=매수체결, 음수=매도체결
        - FID 228: 체결강도 (매수/매도 압력 비율)
        """
        # TODO: 내일 실제 FID 파싱 로직 구현
        stream_type = data.get("type", "")
        print(f"📨 수신 [{stream_type}]: {str(data)[:100]}...")

    def run(self, stock_codes: list):
        """동기 진입점 - asyncio 이벤트 루프 실행"""
        asyncio.run(self.listen(stock_codes))
