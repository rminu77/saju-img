## 사주 이미지 생성 앱

Streamlit 기반의 웹 앱으로, 사용자가 입력한 사주 콘텐츠를 바탕으로  
총운/월별 운세 HTML 리포트와 AI 생성 이미지를 한 번에 만들어 줍니다.  
Google Gemini와 OpenAI API를 동시에 활용해 텍스트·이미지를 생성하며,  
부적 이미지는 제공된 캐릭터 PNG를 편집해 전통 부적 스타일로 변환합니다.

---

### 주요 기능
- 19개 섹션 입력 또는 CSV 업로드로 사주 콘텐츠 자동 로드
- 총운 요약, 핵심 문장, 월별 운세 등 Tailwind 기반 HTML 템플릿 생성
- 사주 대표 이미지를 OpenAI `gpt-image-1`로 생성
- 나나/뱐냐/앙몬드 캐릭터를 이용한 3종 부적 이미지 동시 생성(병렬 처리)
- 생성된 HTML 미리보기, 다운로드, 이미지 표시 및 저장
- 채팅방 요약, 핵심 장면 요약 등 텍스트 생성 보조 기능

---

### 요구 사항
- Python 3.9 이상
- `pip install -r requirements.txt`
- 환경 변수 (`.env` 권장)
  - `GOOGLE_API_KEY`
  - `OPENAI_API_KEY`

---

### 실행 방법
```bash
cd /Users/mason/Documents/사주
python -m venv .venv
source .venv/bin/activate  # Windows는 .venv\Scripts\activate
pip install -r requirements.txt

streamlit run sajun_img.py
```

---

### CSV 업로드 규칙
- `sample_data.csv` 참고
- 첫 행은 `항목,내용`
- `핵심포인트(새해신수)` 등 19개 키가 정확히 일치해야 입력창 자동 채움
- `이름`, `성별`, `생년월일` 항목도 지원

---

### 기타 파일 안내
- `img/`: 부적 변환용 캐릭터 PNG 3종
- `docs/`: 과거 HTML 산출물
- `extracted_sample_data.json`: 전체 샘플 데이터셋
- `render.yaml`, `runtime.txt`: Render 배포 설정 템플릿

---

### Trouble Shooting
- **ThreadPoolExecutor 경고**  
  `missing ScriptRunContext` 메시지는 병렬 처리 중 `st.write` 호출 시 나타나는 Streamlit 경고로, 기능에는 영향이 없습니다.
- **이미지 생성 지연/중단**  
  최근 버전에서는 각 단계별 상태 로그와 예외 처리를 추가했습니다. 콘솔/앱 로그를 확인해 문제 구간을 파악한 뒤 API 키/쿼터/프롬프트를 점검하세요.


