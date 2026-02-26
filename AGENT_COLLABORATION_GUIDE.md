# AI 에이전트 협업 가이드 (Jules & Antigravity)

이 문서는 AI 투자 시스템 프로젝트에서 **Jules(메인 코더)**와 **Antigravity(오케스트레이터)**가 효율적으로 협업하기 위한 규칙을 정의합니다.

## 🤖 역할 분담 (Role Definition)

### 1. Jules (jules.google.com)
- **역할**: 메인 설계자 및 코드 작성자 (Big Brain)
- **주요 임무**: 
    - 복잡한 비즈니스 로직 설계 및 구현
    - 대규모 리팩토링 및 아키텍처 제안
- **협업 방식**: 고도로 정제된 요구사항(프롬프트)을 입력받아 코드를 생성합니다.

### 2. Antigravity (로컬 에이전트)
- **역할**: 운영자, 리뷰어, 테스트 자동화 (Orchestrator)
- **주요 임무**:
    - Jules가 작성한 코드의 로컬 환경 테스트 및 실시간 검증
    - 터미널 실행 로그 수집 및 에러 요약 (Context 최적화)
    - GitHub 커밋 및 DB 관리 자동화
- **협업 방식**: Jules의 코드를 검토하고, 오류 발생 시 핵심 로그만 추려 Jules에게 재요청합니다.

---

## ⚙️ 협업 프로세스 (Workflow)

1. **설계 및 코딩**: 사용자가 Jules에게 개발 목표를 전달합니다.
2. **코드 반영**: Jules가 생성한 코드를 로컬 환경에 저장합니다.
3. **검증 (Antigravity)**: 
    - Antigravity가 `pytest` 또는 실행 스크립트를 통해 코드를 검증합니다.
    - 실패 시, Antigravity가 **로컬 LLM(Ollama)**을 사용해 로그를 500자 이내로 요약합니다.
4. **리드백**: 요약된 로그를 Jules에게 전달하여 수정을 요청합니다 (토큰 절약 전략).
5. **완료 및 동기화**: 검증 완료된 코드는 Antigravity가 GitHub (`AI_STUDY_WH`)에 커밋합니다.

---

## 🛡️ 보안 및 금기 사항 (Rules)

- **.env 파일 절대 엄금**: API 키나 비밀번호가 포함된 `.env` 파일은 절대 GitHub에 올리지 않습니다. (Antigravity가 `git add` 시 상시 감시)
- **최대 Context 유지**: Jules에게 파일 전체를 던지지 않고, 수정이 필요한 특정 함수나 클래스 단위로 대화하여 비용을 최적화합니다.
- **Single Source of Truth**: 모든 최종 코드는 GitHub `main` 브랜치를 기준으로 합니다.

---

## 📂 주요 파일 구조 및 관리
- `/data`: 백테스트 결과 및 시계열 데이터 저장 (Docker DB와 연동)
- `/src`: 핵심 투자 로직 및 API 연계 코드
- `/tests`: Antigravity가 상시 실행할 유닛 테스트 코드
