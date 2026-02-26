import time
import subprocess
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# 감시할 설정
WATCH_DIRECTORY = "."
RUN_COMMAND = "python test_gemini.py"  # 나중에 백테스트 실행 스크립트로 변경 가능
LOG_FILE = "execution_log.txt"

class AutoRunnerHandler(FileSystemEventHandler):
    def on_modified(self, event):
        # .py 파일이 수정되었을 때만 실행 (로그 파일이나 .venv 제외)
        if event.src_path.endswith(".py") and not "auto_pilot.py" in event.src_path:
            print(f"\n🚀 파일 변경 감지: {event.src_path}")
            print(f"⏳ {RUN_COMMAND} 실행 중...")
            
            with open(LOG_FILE, "a") as log:
                log.write(f"\n--- 실행 시각: {time.ctime()} ---\n")
                try:
                    # 명령어 실행 및 결과 캡처
                    result = subprocess.run(
                        RUN_COMMAND, shell=True, capture_output=True, text=True
                    )
                    log.write(result.stdout)
                    log.write(result.stderr)
                    
                    if result.returncode == 0:
                        print("✅ 실행 성공! 결과를 execution_log.txt에서 확인하세요.")
                    else:
                        print("❌ 실행 실패! 에러 로그를 확인하세요.")
                        
                except Exception as e:
                    log.write(f"시스템 에러: {str(e)}\n")
                    print(f"❌ 시스템 에러 발생: {str(e)}")

if __name__ == "__main__":
    event_handler = AutoRunnerHandler()
    observer = Observer()
    observer.schedule(event_handler, WATCH_DIRECTORY, recursive=False)
    
    print(f"👀 {WATCH_DIRECTORY} 폴더 감시 중... (코드 수정 시 자동 실행)")
    print("💡 종료하려면 Ctrl+C를 누르세요.")
    
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
