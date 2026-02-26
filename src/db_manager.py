import os
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()

class DBManager:
    def __init__(self):
        self.conn_params = {
            "host": "localhost",
            "database": "ai_invest",
            "user": "myuser",
            "password": "mypassword",
            "port": "5432"
        }

    def get_connection(self):
        try:
            return psycopg2.connect(**self.conn_params)
        except Exception as e:
            print(f"❌ DB 연결 실패: {e}")
            return None

    def save_backtest_result(self, strategy_name, params, result_data):
        """
        백테스트 결과를 DB에 저장합니다.
        params: dict (인자값), result_data: dict (수익률 등 결과)
        """
        conn = self.get_connection()
        if not conn: return
        
        try:
            with conn.cursor() as cur:
                # 테이블이 없으면 생성 (TimescaleDB 하이퍼테이블 고려 가능)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS backtest_results (
                        id SERIAL PRIMARY KEY,
                        execution_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        strategy_name TEXT,
                        parameters JSONB,
                        results JSONB
                    );
                """)
                
                cur.execute("""
                    INSERT INTO backtest_results (strategy_name, parameters, results)
                    VALUES (%s, %s, %s)
                """, (strategy_name, json.dumps(params), json.dumps(result_data)))
                
            conn.commit()
            print(f"✅ 백테스트 결과 저장 완료: {strategy_name}")
        except Exception as e:
            print(f"❌ 데이터 저장 실패: {e}")
            conn.rollback()
        finally:
            conn.close()

import json # 스크립트 내에서 json 사용을 위해 추가
