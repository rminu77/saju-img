# saju_image_app.py
# pip install streamlit google-genai pillow python-dotenv

import streamlit as st
from google import genai
from PIL import Image
from io import BytesIO
import time
import os
import base64
from dotenv import load_dotenv
import requests
from typing import Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

load_dotenv()

st.set_page_config(page_title="사주 → HTML 생성기", page_icon="🧧", layout="wide")

# ----------------------------
# 로그인 체크
# ----------------------------
def check_login():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        st.title("🔐 로그인")
        st.text_input("ID")
        password = st.text_input("PW", type="password")

        if st.button("로그인"):
            if password == "mateplan":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("비밀번호가 올바르지 않습니다.")
        st.stop()

check_login()

# ----------------------------
# 설정
# ----------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TEXT_MODEL = "gemini-2.5-pro"                 # 프롬프트 작성 모델
IMAGE_MODEL = "gemini-2.5-flash-image-preview"  # 이미지 생성 모델
OPENAI_TEXT_MODEL = "gpt-4.1-mini"  # 장면 요약 모델
OPENAI_IMAGE_MODEL = "gpt-image-1"
OPENAI_IMAGE_SIZE = "1024x1024"
# 현재 스크립트 위치 기준으로 result 디렉토리 설정
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")
DEFAULT_SYSTEM_INSTRUCTION = (
    "A mystical, hopeful scene rooted in Korean culture. "
    "Draw the characters in a way that highlights their personality, similar to Disney's Tangled and Encanto. "
    "The overall scene should be bright, rich in color, and vibrant, must have no wrinkles, with a lovely emphasis on the characters. "
    "Express the faces in a Ghibli style. The lighting should be soft but powerful, and the characters should embody both warmth and vitality. "
    "The atmosphere should be both fantastical and dramatic."
)
DEFAULT_SUMMARY_INSTRUCTION = (
    "You are a Korean-to-English creative synthesis assistant with a warm, hopeful tone. "
    "Read the provided Korean saju text and create a vivid, single-scene description that can be rendered as one beautiful painting. "
    "Your description MUST include: "
    "1. WHO: A specific human figure (describe gender, youthful for their age, beautiful, and elegant appearance, attire, posture) "
    "2. WHERE: A background that depicts the saju's contents "
    "3. WHAT: A specific action or gesture the person is performing in that moment "
    "The background must always be in Korea and include Korean cultural elements. Women wear a skirt hanbok, men wear pants hanbok.) "
    "ALWAYS center the description around the human figure - describe what the person looks like, what they are doing, and where they are. "
    "Portray the human figure as youthful for their age, beautiful, dignified, and elegant. "
    "Focus on positive, uplifting, and hopeful visual metaphors that inspire optimism and growth. "
    "Even when addressing challenges, frame them as opportunities for transformation and renewal. "
    "Emphasize bright colors, ascending movements, blooming elements, and harmonious compositions. "
    "Focus on concrete visual motifs and atmospheric cues that evoke hope and possibility. "
    "Create a description that an artist can immediately visualize and paint as a single, cohesive scene. "
    "Output the description in English as 1-2 sentences."
)

# ----------------------------
# 유틸
# ----------------------------
def get_gemini_client():
    if not GEMINI_API_KEY:
        return None
    try:
        return genai.Client(api_key=GEMINI_API_KEY)
    except Exception:
        return None

def get_openai_client():
    if not OPENAI_API_KEY or not OpenAI:
        return None
    try:
        # httpx 클라이언트를 명시적으로 생성하여 프록시 문제 우회
        # trust_env=False로 환경 변수의 프록시 설정을 무시
        import httpx
        http_client = httpx.Client(trust_env=False)
        client = OpenAI(api_key=OPENAI_API_KEY, http_client=http_client)
        return client
    except ImportError:
        # httpx를 사용할 수 없는 경우 기본 방식으로 시도
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)
            return client
        except Exception as e:
            st.warning(f"OpenAI 클라이언트 초기화 실패: {e}")
            return None
    except Exception as e:
        st.warning(f"OpenAI 클라이언트 초기화 실패: {e}")
        return None


def convert_tone_to_dosa(
    source_text: str,
    user_name: str,
    openai_client: Optional[OpenAI] = None,
) -> str:
    """
    입력 텍스트의 말투를 도사 스타일로 변환
    """
    system_instruction = f"""당신은 도사 말투로 변환하는 전문가입니다.

변환 규칙:
- 반말만 사용
- 밝고 유쾌하되 도사다운 무게와 신비감 유지
- 시작구를 교차 사용: "어디보자…", "오호…", "옳거니!", "이거 참 묘하구나", "허허, 재밌네…"
- 끝맺음: "~하네", "~이니라", "잊지 말게", "어떤가?"
- 가끔 부채 이모지 🪭 사용
- 사용자를 항상 "{user_name}"(으)로 부름
- 내용은 절대 요약하지 말고 원문의 의미를 모두 살려서 말투만 변환
- 원문의 구조와 문단을 그대로 유지"""

    user_msg = f"""다음 텍스트를 도사 말투로 변환해주세요. 내용은 절대 줄이지 말고 말투만 바꿔주세요:

{source_text}"""

    if not openai_client:
        raise ValueError("OpenAI 클라이언트가 초기화되지 않았습니다.")

    completion = openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_msg},
        ]
    )
    return (completion.choices[0].message.content or "").strip()

def summarize_for_visuals(
    source_text: str,
    provider: str = "gemini",
    gemini_client: Optional[genai.Client] = None,
    openai_client: Optional[OpenAI] = None,
    system_instruction: str = DEFAULT_SUMMARY_INSTRUCTION,
    openai_text_model: str = OPENAI_TEXT_MODEL,
) -> str:
    """
    사주 텍스트를 그림을 위한 1~2개의 핵심 문장으로 요약.
    """
    user_msg = f"""
[SAJU TEXT / Korean]
{source_text}

[REQUEST]
- Summarize into one or two sentences highlighting visual motifs, elements, and atmosphere for illustration.
- Keep it concrete and metaphorical, avoid fortune-telling claims.
"""
    if provider == "openai":
        if not openai_client:
            raise ValueError("OpenAI 클라이언트가 초기화되지 않았습니다.")
        completion = openai_client.chat.completions.create(
            model=openai_text_model,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_msg},
            ]
        )
        return (completion.choices[0].message.content or "").strip()

    if not gemini_client:
        raise ValueError("Gemini 클라이언트가 초기화되지 않았습니다.")

    resp = gemini_client.models.generate_content(
        model=TEXT_MODEL,
        contents=[system_instruction, user_msg]
    )
    return (resp.text or "").strip()

def write_prompt_from_saju(
    source_text: str,
    system_instruction: str = DEFAULT_SYSTEM_INSTRUCTION,
    provider: str = "gemini",
    gemini_client: Optional[genai.Client] = None,
    openai_client: Optional[OpenAI] = None,
    core_scene: Optional[str] = None,
    openai_text_model: str = OPENAI_TEXT_MODEL,
) -> str:
    """
    사주 텍스트와 스타일 지시사항을 결합하여 직접 이미지 생성 프롬프트로 반환
    """
    # 기본 스타일 프롬프트로 시작
    prompt_parts = [system_instruction]

    # 핵심 장면 추가
    if core_scene:
        prompt_parts.append(core_scene)

    # 모든 부분을 하나의 프롬프트로 결합
    return " ".join(prompt_parts)

def generate_images(
    prompt: str,
    num_images: int = 3,
    provider: str = "gemini",
    gemini_client: Optional[genai.Client] = None,
    openai_client: Optional[OpenAI] = None,
):
    """
    텍스트만으로 이미지 생성. 최대 num_images장 시도.
    반환: PIL.Image 또는 None의 리스트
    """
    images = []
    if provider == "openai":
        if not openai_client:
            return [None] * num_images
        for _ in range(num_images):
            try:
                response = openai_client.images.generate(
                    model=OPENAI_IMAGE_MODEL,
                    prompt=prompt,
                    size=OPENAI_IMAGE_SIZE,
                    n=1,
                )
                img_data = response.data[0] if response.data else None
                img_bytes = None
                if getattr(img_data, "b64_json", None):
                    img_bytes = base64.b64decode(img_data.b64_json)
                elif getattr(img_data, "url", None):
                    img_bytes = requests.get(img_data.url).content

                img = Image.open(BytesIO(img_bytes)).convert("RGBA") if img_bytes else None
                images.append(img)
            except Exception:
                images.append(None)
        return images

    if not gemini_client:
        return [None] * num_images

    for _ in range(num_images):
        try:
            response = gemini_client.models.generate_content(
                model=IMAGE_MODEL,
                contents=f"Create a picture of: {prompt}"
            )

            # google-genai 응답에서 이미지 추출
            img = None
            if getattr(response, "candidates", None):
                parts = response.candidates[0].content.parts
                for part in parts:
                    # part.inline_data.data 가 바이너리 이미지
                    if getattr(part, "inline_data", None) and getattr(part.inline_data, "data", None):
                        data = part.inline_data.data
                        img = Image.open(BytesIO(data))
                        break
            images.append(img)
        except Exception:
            images.append(None)
    return images

def generate_html(user_name: str, gender: str, solar_date: str, lunar_date: str,
                  birth_time: str, sections: dict, image_base64: str) -> str:
    """
    19개 섹션 내용을 받아서 HTML을 생성
    image_base64: base64로 인코딩된 이미지 데이터
    """
    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{user_name} 님의 토정비결</title>
    <!-- Tailwind CSS CDN -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Google Fonts: Inter and Noto Sans KR -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Noto+Sans+KR:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        body {{
            font-family: 'Inter', 'Noto Sans KR', sans-serif;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }}
    </style>
</head>
<body class="bg-gray-100 py-10 px-4">

    <main class="max-w-3xl mx-auto bg-white shadow-2xl rounded-xl overflow-hidden">
        <div class="p-8 sm:p-12">

            <h1 class="text-3xl sm:text-4xl font-bold text-gray-800 mb-4 text-center">
                {user_name} 님의 토정비결
            </h1>

            <p class="text-lg text-gray-600 mb-10 font-medium text-center">
                <strong>[ {gender} ]</strong> 양력 {solar_date} {birth_time} / 음력 {lunar_date} {birth_time}
            </p>

            <!-- 섹션: 그림으로 보는 새해운세 -->
            <section class="mb-10">
                <h2 class="text-2xl font-semibold text-blue-700 border-b-2 border-blue-100 pb-3 mb-6">
                    그림으로 보는 새해운세
                </h2>
                <div class="flex justify-center">
                    <img src="data:image/png;base64,{image_base64}" alt="새해운세 이미지" class="rounded-lg shadow-lg max-w-full h-auto">
                </div>
            </section>
"""

    # 19개 섹션 추가
    section_titles = [
        "핵심포인트", "올해의총운", "일년신수(전반기)", "일년신수(후반기)",
        "올해의연애운", "올해의건강운", "올해의직장운", "올해의소망운",
        "올해의여행이사운", "월별운", "재물운의특성", "재물모으는법",
        "재물손실막는법", "현재의재물운", "시기적운세", "대길",
        "대흉", "현재의길흉사", "운명뛰어넘기"
    ]

    for title in section_titles:
        content = sections.get(title, "").strip()
        if content:
            html += f"""
            <section class="mb-10">
                <h2 class="text-2xl font-semibold text-blue-700 border-b-2 border-blue-100 pb-3 mb-6">
                    {title}
                </h2>
                <div class="space-y-4">
                    <p class="text-base text-gray-700 leading-relaxed">
                        {content.replace(chr(10), '<br>')}
                    </p>
                </div>
            </section>
"""

    html += """
        </div>
    </main>

</body>
</html>
"""
    return html

# ----------------------------
# UI
# ----------------------------
st.title("🧧 토정비결 HTML 생성기")
st.caption("19개 항목을 입력하면 이미지와 함께 HTML을 생성합니다")

# result 디렉토리가 없으면 생성
if not os.path.exists(RESULT_DIR):
    try:
        os.makedirs(RESULT_DIR)
    except Exception as e:
        st.warning(f"result 디렉토리 생성 실패: {e}. 파일 저장은 건너뜁니다.")

gemini_client = get_gemini_client()
openai_client = get_openai_client()
openai_available = bool(openai_client)

if not openai_available:
    st.error("OPENAI_API_KEY가 설정되지 않았거나 openai 패키지가 없습니다.")
    st.stop()

if "core_scene_summary" not in st.session_state:
    st.session_state.core_scene_summary = ""
if "generated_html" not in st.session_state:
    st.session_state.generated_html = None
if "generated_image" not in st.session_state:
    st.session_state.generated_image = None
if "html_filename" not in st.session_state:
    st.session_state.html_filename = None

# 사용자 정보 입력
st.subheader("📋 기본 정보")
col1, col2 = st.columns(2)
with col1:
    user_name = st.text_input("이름", value="김영희")
    gender = st.selectbox("성별", ["남자", "여자"])
with col2:
    solar_date = st.text_input("양력 생년월일", value="1988-08-09")
    lunar_date = st.text_input("음력 생년월일", value="1988-06-27")
    birth_time = st.text_input("시간", value="辰時")

st.markdown("---")
st.subheader("📝 19개 항목 입력")

# 19개 입력창
sections = {}
section_titles = [
    "핵심포인트", "올해의총운", "일년신수(전반기)", "일년신수(후반기)",
    "올해의연애운", "올해의건강운", "올해의직장운", "올해의소망운",
    "올해의여행이사운", "월별운", "재물운의특성", "재물모으는법",
    "재물손실막는법", "현재의재물운", "시기적운세", "대길",
    "대흉", "현재의길흉사", "운명뛰어넘기"
]

for title in section_titles:
    sections[title] = st.text_area(title, height=100, key=title)

system_prompt_input = st.text_area(
    "이미지 생성 시스템 프롬프트",
    value=DEFAULT_SYSTEM_INSTRUCTION,
    height=120,
    help="이미지 프롬프트 작성 모델에 전달할 시스템 메시지입니다.",
)
system_prompt = system_prompt_input if system_prompt_input.strip() else DEFAULT_SYSTEM_INSTRUCTION

summary_prompt_input = st.text_area(
    "장면요약 시스템 프롬프트",
    value=DEFAULT_SUMMARY_INSTRUCTION,
    height=120,
    help="핵심 장면 요약 생성 모델에 전달할 시스템 메시지입니다.",
)
summary_prompt = summary_prompt_input if summary_prompt_input.strip() else DEFAULT_SUMMARY_INSTRUCTION

st.markdown("---")
generate = st.button("🚀 HTML 생성", type="primary", use_container_width=True)

if generate:
    # "올해의총운" 텍스트로 이미지 생성
    base_text = sections.get("올해의총운", "").strip()
    if not base_text:
        st.error("'올해의총운'을 입력해주세요. 이 내용으로 이미지를 생성합니다.")
        st.stop()

    # 이미지 생성 시작 시점의 설정을 고정
    locked_system_prompt = system_prompt
    locked_summary_prompt = summary_prompt
    locked_openai_client = openai_client

    with st.spinner("🔍 핵심 장면 추출 중 (gpt-4.1-mini 사용)..."):
        try:
            core_scene = summarize_for_visuals(
                base_text,
                provider="openai",
                gemini_client=None,
                openai_client=locked_openai_client,
                system_instruction=locked_summary_prompt,
                openai_text_model="gpt-4.1-mini",
            )
        except Exception as exc:
            st.error(f"핵심 장면 요약 생성 중 오류가 발생했습니다: {exc}")
            st.stop()

    core_scene = (core_scene or "").strip()
    st.session_state["core_scene_summary"] = core_scene
    if core_scene:
        st.markdown("#### ✨ 핵심 장면 요약")
        st.write(core_scene)

    with st.spinner("📝 프롬프트 작성 중..."):
        try:
            prompt = write_prompt_from_saju(
                base_text,
                system_instruction=locked_system_prompt,
                provider="openai",
                gemini_client=None,
                openai_client=locked_openai_client,
                core_scene=core_scene,
                openai_text_model="gpt-4.1-mini",
            )
        except Exception as exc:
            st.error(f"프롬프트 생성 중 오류가 발생했습니다: {exc}")
            st.stop()

    if not prompt:
        st.error("프롬프트 생성에 실패했습니다. 입력 내용을 다시 확인해주세요.")
        st.stop()

    final_prompt = prompt

    # 이미지 생성과 말투 변환을 병렬로 실행
    with st.spinner("🎨 이미지 생성 및 🪭 말투 변환 중..."):
        # 병렬 실행을 위한 함수들
        def generate_image_task():
            imgs = generate_images(
                final_prompt,
                num_images=1,
                provider="openai",
                gemini_client=None,
                openai_client=locked_openai_client,
            )
            return imgs

        def convert_tone_task():
            converted = {}
            for title, content in sections.items():
                if content.strip():
                    try:
                        converted[title] = convert_tone_to_dosa(
                            source_text=content,
                            user_name=user_name,
                            openai_client=locked_openai_client
                        )
                    except Exception as e:
                        st.warning(f"'{title}' 말투 변환 실패: {e}, 원문 사용")
                        converted[title] = content
                else:
                    converted[title] = content
            return converted

        # 병렬 실행
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_image = executor.submit(generate_image_task)
            future_tone = executor.submit(convert_tone_task)

            # 결과 수집
            imgs = future_image.result()
            converted_sections = future_tone.result()

    # 이미지 처리
    valid = [i for i in imgs if i is not None]
    if not valid:
        st.error("이미지 생성에 실패했습니다.")
        st.stop()

    # 이미지를 base64로 인코딩
    img = valid[0]
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    # 이미지 파일도 저장 (로컬 백업용)
    timestamp = int(time.time())
    image_filename = f"saju_generated_{timestamp}.png"

    # 파일 저장 시도 (실패해도 계속 진행)
    try:
        image_path = os.path.join(RESULT_DIR, image_filename)
        img.save(image_path, format="PNG")
    except Exception as e:
        pass  # 파일 저장 실패는 무시

    # HTML 생성
    with st.spinner("📄 HTML 생성 중..."):
        html_content = generate_html(
            user_name=user_name,
            gender=gender,
            solar_date=solar_date,
            lunar_date=lunar_date,
            birth_time=birth_time,
            sections=converted_sections,
            image_base64=img_base64
        )

        html_filename = f"{user_name}_tojeung_{timestamp}.html"

        # 파일 저장 시도 (실패해도 계속 진행)
        try:
            html_path = os.path.join(RESULT_DIR, html_filename)
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)
        except Exception as e:
            pass  # 파일 저장 실패는 무시

    # 세션 상태에 결과 저장
    st.session_state.generated_html = html_content
    st.session_state.generated_image = img
    st.session_state.html_filename = html_filename

    st.success(f"✅ 이미지 생성 및 말투 변환 완료!")

# 결과물 표시 (세션 상태에서 가져옴)
if st.session_state.generated_html is not None:
    st.markdown("---")
    st.markdown("### 🎨 생성 결과")

    # 이미지 표시
    if st.session_state.generated_image is not None:
        st.image(st.session_state.generated_image, caption="생성된 이미지", use_container_width=True)

    # HTML 다운로드 버튼
    st.download_button(
        label="📥 HTML 다운로드",
        data=st.session_state.generated_html,
        file_name=st.session_state.html_filename,
        mime="text/html",
        use_container_width=True
    )

    # HTML 미리보기 (항상 표시)
    st.markdown("---")
    st.markdown("### 📄 HTML 미리보기")
    st.components.v1.html(st.session_state.generated_html, height=800, scrolling=True)

if not generate:
    summary_display = st.session_state.get("core_scene_summary", "").strip()
    if summary_display:
        st.markdown("#### ✨ 핵심 장면 요약")
        st.write(summary_display)
