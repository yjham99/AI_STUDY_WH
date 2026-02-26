"""
외부 정보 통합 허브 (Info Aggregator)
- 네이버 증권, 야후 파이낸스, 유튜브 키워드(3프로TV 등), 전략 게시판
- 수집된 정보를 ai_invest DB의 market_intelligence 테이블에 저장
- 일일 Gemini API 브리핑 자동 요약 생성
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

class InfoAggregator:
    """외부 정보 수집 및 DB 저장 허브"""

    def get_naver_news(self, stock_code: str) -> list:
        """
        네이버 증권 뉴스 수집 (RSS 파싱)
        - 종목코드 기반 뉴스 제목, 링크, 발행일 수집
        """
        # TODO: 종목별 네이버 증권 뉴스 RSS 파싱 구현
        url = f"https://finance.naver.com/item/news_news.nhn?code={stock_code}"
        print(f"📰 네이버 증권 뉴스 수집 중: {stock_code}")
        return []

    def get_yahoo_finance(self, ticker: str) -> dict:
        """
        야후 파이낸스 (yfinance 라이브러리)
        - 해외 지수(S&P500, NASDAQ), 환율(USD/KRW) 등 매크로 데이터
        """
        try:
            import yfinance as yf
            data = yf.Ticker(ticker)
            info = data.info
            print(f"📈 야후 파이낸스 데이터 수집 완료: {ticker}")
            return info
        except ImportError:
            print("⚠️ yfinance 미설치: pip install yfinance")
            return {}

    def get_youtube_keywords(self, channel_keywords: list = ["3프로TV"]) -> list:
        """
        YouTube Data API v3를 통한 키워드 수집
        - 채널 최신 영상 제목·설명에서 종목·전략 키워드 추출
        - 감성 분석(Gemini)에 Feed
        """
        # TODO: YouTube Data API v3 연동 구현
        # API Key는 .env의 YOUTUBE_API_KEY에 추가 필요
        youtube_api_key = os.getenv("YOUTUBE_API_KEY", "")
        if not youtube_api_key:
            print("⚠️ YOUTUBE_API_KEY가 .env에 없습니다.")
            return []
        print(f"🎬 유튜브 키워드 수집 중: {channel_keywords}")
        return []

    def save_to_db(self, source: str, data: dict):
        """
        수집된 정보를 market_intelligence 테이블에 저장
        """
        # TODO: db_manager.py 활용하여 저장 구현
        print(f"💾 외부 정보 DB 저장: [{source}]")

    def generate_daily_briefing(self, data: dict) -> str:
        """
        Gemini API로 일일 시장 브리핑 요약 자동 생성
        """
        from google import genai
        api_key = os.getenv("GEMINI_API_KEY_TIER1")
        client = genai.Client(api_key=api_key)
        prompt = f"다음 시장 데이터를 바탕으로 오늘의 투자 전략 브리핑을 200자 이내로 요약해줘: {str(data)}"
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        return response.text


if __name__ == "__main__":
    agg = InfoAggregator()
    agg.get_yahoo_finance("^IXIC")  # NASDAQ
    agg.get_naver_news("005930")    # 삼성전자
