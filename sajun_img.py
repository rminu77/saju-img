# saju_image_app.py
# pip install streamlit google-genai pillow python-dotenv beautifulsoup4

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

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

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
DEFAULT_CHAT_SUMMARY_INSTRUCTION = """당신은 도사 말투로 사주를 요약하는 전문가입니다.

변환 규칙:
- 반말만 사용
- 밝고 유쾌하되 도사다운 무게와 신비감 유지
- 다음과 같은 표현을 적절히 사용: "어디보자…", "오호…", "옳거니!", "이거 참 묘하구나", "허허, 재밌네…", "~하네", "~이니라", "잊지 말게", "어떤가?"
- 가끔 부채 이모지 🪭 사용
- 사용자를 항상 "{user_name}"(으)로 부름
- 4500자 내외로 요약 (최대 5000자)
- 핵심 내용을 빠짐없이 전달하되 도사스러운 표현으로 재구성
- - 맨 마지막에 더 자세히 보려면 토정비결 보기 버튼을 눌러보라고 안내해"""

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
- 다음과 같은 표현을 적절히 사용: "어디보자…", "오호…", "옳거니!", "이거 참 묘하구나", "허허, 재밌네…", "~하네", "~이니라", "잊지 말게", "어떤가?"
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
    gender: str = "여자",
) -> str:
    """
    사주 텍스트를 그림을 위한 1~2개의 핵심 문장으로 요약.
    """
    user_msg = f"""
[SAJU TEXT / Korean]
{source_text}

[GENDER]
{gender}

[REQUEST]
- Summarize into one or two sentences highlighting visual motifs, elements, and atmosphere for illustration.
- Keep it concrete and metaphorical, avoid fortune-telling claims.
- The main character should be a {gender} ({"woman" if gender == "여자" else "man"}).
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
        /* 'Inter' 폰트를 기본으로 하되, 한글은 'Noto Sans KR'을 사용하도록 설정합니다. */
        body {{
            font-family: 'Inter', 'Noto Sans KR', sans-serif;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }}
    </style>
</head>
<body class="bg-gray-100 py-10 px-4">

    <!-- 메인 콘텐츠 카드 -->
    <main class="max-w-3xl mx-auto bg-white shadow-2xl rounded-xl overflow-hidden">
        <div class="p-8 sm:p-12">

            <!-- 제목 -->
            <h1 class="text-3xl sm:text-4xl font-bold text-gray-800 mb-4 text-center">
                {user_name} 님의 토정비결
            </h1>

            <!-- 사용자 정보 -->
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

    # 섹션별 색상 정의
    section_colors = {
        "핵심포인트": ("blue", "blue"),
        "올해의총운": ("blue", "blue"),
        "일년신수(전반기": ("blue", "blue"),
        "일년신수(후반기": ("blue", "blue"),
        "올해의연애운": ("pink", "pink"),
        "올해의건강운": ("green", "green"),
        "올해의직장운": ("purple", "purple"),
        "올해의소망운": ("indigo", "indigo"),
        "올해의여행이사운": ("teal", "teal"),
        "월별운": ("orange", "orange"),
        "재물운의특성": ("yellow", "yellow"),
        "재물모으는법": ("yellow", "yellow"),
        "재물손실막는법": ("yellow", "yellow"),
        "현재의재물운": ("yellow", "yellow"),
        "시기적운세": ("red", "red"),
        "대길대흉": ("gray", "gray"),  # 대길대흉은 회색 테두리
        "현재의길흉사": ("cyan", "cyan"),
        "운명뛰어넘기": ("violet", "violet")
    }

    section_display_titles = {
        "핵심포인트": "핵심포인트",
        "올해의총운": "올해의 총운",
        "일년신수(전반기": "일년신수(전반기)",
        "일년신수(후반기": "일년신수(후반기)",
        "올해의연애운": "올해의 연애운",
        "올해의건강운": "올해의 건강운",
        "올해의직장운": "올해의 직장운",
        "올해의소망운": "올해의 소망운",
        "올해의여행이사운": "올해의 여행·이사운",
        "월별운": "월별운",
        "재물운의특성": "재물운의 특성",
        "재물모으는법": "재물 모으는 법",
        "재물손실막는법": "재물 손실 막는 법",
        "현재의재물운": "현재의 재물운",
        "시기적운세": "시기적 운세",
        "대길대흉": "대길대흉",  # 대길과 대흉을 하나의 섹션으로 통합
        "현재의길흉사": "현재의 길흉사",
        "운명뛰어넘기": "운명 뛰어넘기"
    }

    for key, display_title in section_display_titles.items():
        # 대길과 대흉은 개별적으로 스킵 (대길대흉 섹션에서 처리)
        if key in ["대길", "대흉"]:
            continue

        content = sections.get(key, "").strip()
        # 대길대흉 섹션은 대길이나 대흉 중 하나라도 있으면 표시
        if key == "대길대흉":
            if not sections.get("대길", "").strip() and not sections.get("대흉", "").strip():
                continue
        elif not content:
            continue

        # 색상 가져오기
        color = section_colors.get(key, ("blue", "blue"))

        html += f"""
            <!-- 섹션: {display_title} -->
            <section class="mb-10">
                <h2 class="text-2xl font-semibold text-{color[0]}-700 border-b-2 border-{color[1]}-100 pb-3 mb-6">
                    {display_title}
                </h2>
                """

        # 월별운은 특별 처리 (그리드 레이아웃)
        if key == "월별운":
            # 월별 정보 파싱
            months = []
            lines = content.split('\n')
            current_month = None
            current_text = []

            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # "01월", "1월" 등의 패턴 찾기
                if line.endswith('월') and len(line) <= 4:
                    # 이전 월 데이터 저장
                    if current_month and current_text:
                        months.append({'month': current_month, 'text': ' '.join(current_text)})
                    current_month = line
                    current_text = []
                else:
                    current_text.append(line)

            # 마지막 월 저장
            if current_month and current_text:
                months.append({'month': current_month, 'text': ' '.join(current_text)})

            # 그리드 레이아웃으로 출력
            html += '                <div class="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-6">\n'
            for month_data in months:
                html += '                    <div class="bg-gray-50 p-4 rounded-lg">\n'
                html += f'                        <h4 class="text-lg font-bold text-gray-800 mb-1">{month_data["month"]}</h4>\n'
                html += f'                        <p class="text-base text-gray-700 leading-relaxed">{month_data["text"]}</p>\n'
                html += '                    </div>\n'
            html += '                </div>\n'
        # 대길대흉 섹션 특별 처리
        elif display_title == "대길대흉":
            # 대길과 대흉 내용을 분리
            daegil_content = sections.get("대길", "").strip()
            daeheung_content = sections.get("대흉", "").strip()

            # 대길 박스
            if daegil_content:
                html += '                <!-- 대길 -->\n'
                html += '                <div class="mb-8 p-6 bg-blue-50 rounded-lg border border-blue-200">\n'
                html += '                    <h3 class="text-2xl font-bold text-blue-800 mb-4">\n'
                html += '                        대길 (大吉)\n'
                html += '                    </h3>\n'
                html += '                    <div class="space-y-4">\n'

                # 대길 내용 파싱
                paragraphs = [p.strip() for p in daegil_content.split('\n\n') if p.strip()]
                for para in paragraphs:
                    lines = [l.strip() for l in para.split('\n') if l.strip()]
                    if len(lines) > 1 and len(lines[0]) < 100:
                        # h4 제목 + 여러 p
                        html += '                        <div>\n'
                        html += f'                            <h4 class="text-lg font-semibold text-gray-700 mb-1">{lines[0]}</h4>\n'
                        for i, line in enumerate(lines[1:]):
                            if i == 0:
                                html += f'                            <p class="text-base text-gray-700 leading-relaxed">{line}</p>\n'
                            else:
                                html += f'                            <p class="text-base text-gray-700 leading-relaxed mt-4">{line}</p>\n'
                        html += '                        </div>\n'
                    else:
                        # p만
                        for line in lines:
                            html += f'                        <p class="text-base text-gray-700 leading-relaxed">{line}</p>\n'

                html += '                    </div>\n'
                html += '                </div>\n'
                html += '\n'

            # 대흉 박스
            if daeheung_content:
                html += '                <!-- 대흉 -->\n'
                html += '                <div class="p-6 bg-red-50 rounded-lg border border-red-200">\n'
                html += '                    <h3 class="text-2xl font-bold text-red-800 mb-4">\n'
                html += '                        대흉 (大凶)\n'
                html += '                    </h3>\n'
                html += '                    <div class="space-y-4">\n'

                # 대흉 내용 파싱
                paragraphs = [p.strip() for p in daeheung_content.split('\n\n') if p.strip()]
                for para in paragraphs:
                    lines = [l.strip() for l in para.split('\n') if l.strip()]
                    if len(lines) > 1 and len(lines[0]) < 100:
                        # h4 제목 + 여러 p
                        html += '                        <div>\n'
                        html += f'                            <h4 class="text-lg font-semibold text-gray-700 mb-1">{lines[0]}</h4>\n'
                        for i, line in enumerate(lines[1:]):
                            if i == 0:
                                html += f'                            <p class="text-base text-gray-700 leading-relaxed">{line}</p>\n'
                            else:
                                html += f'                            <p class="text-base text-gray-700 leading-relaxed mt-4">{line}</p>\n'
                        html += '                        </div>\n'
                    else:
                        # p만
                        for line in lines:
                            html += f'                        <p class="text-base text-gray-700 leading-relaxed">{line}</p>\n'

                html += '                    </div>\n'
                html += '                </div>\n'
        else:
            # 일반 섹션 처리
            # 내용을 줄바꿈으로 분리
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]

            # 각 문단을 다시 줄바꿈으로 분리하여 제목과 내용 구분
            formatted_blocks = []
            for para in paragraphs:
                lines = [l.strip() for l in para.split('\n') if l.strip()]
                if len(lines) == 0:
                    continue

                # 첫 줄이 짧고 제목처럼 보이면 h3로 처리
                if len(lines) > 1 and len(lines[0]) < 100:
                    # h3 + 여러 p
                    block = {
                        'type': 'titled',
                        'title': lines[0],
                        'paragraphs': lines[1:]
                    }
                else:
                    # p만
                    block = {
                        'type': 'plain',
                        'paragraphs': lines
                    }
                formatted_blocks.append(block)

            # 첫 블록이 titled면 space-y-6, 아니면 space-y-4
            if formatted_blocks and formatted_blocks[0]['type'] == 'titled':
                html += '                <div class="space-y-6">\n'
                for block in formatted_blocks:
                    if block['type'] == 'titled':
                        html += '                    <div>\n'
                        html += f'                        <h3 class="text-xl font-bold text-gray-700 mb-2">\n'
                        html += f'                            {block["title"]}\n'
                        html += f'                        </h3>\n'
                        for i, para in enumerate(block['paragraphs']):
                            if i == 0:
                                html += f'                        <p class="text-base text-gray-700 leading-relaxed">\n'
                            else:
                                html += f'                        <p class="text-base text-gray-700 leading-relaxed mt-4">\n'
                            html += f'                            {para}\n'
                            html += f'                        </p>\n'
                        html += '                    </div>\n'
                    else:
                        for para in block['paragraphs']:
                            html += f'                    <p class="text-base text-gray-700 leading-relaxed">\n'
                            html += f'                        {para}\n'
                            html += f'                    </p>\n'
                html += '                </div>\n'
            else:
                html += '                <div class="space-y-4 text-base text-gray-700 leading-relaxed">\n'
                for block in formatted_blocks:
                    if block['type'] == 'titled':
                        html += f'                    <h3 class="text-xl font-bold text-gray-700 mb-2">\n'
                        html += f'                        {block["title"]}\n'
                        html += f'                    </h3>\n'
                        for para in block['paragraphs']:
                            html += f'                    <p>\n'
                            html += f'                        {para}\n'
                            html += f'                    </p>\n'
                    else:
                        for para in block['paragraphs']:
                            html += f'                    <p>\n'
                            html += f'                        {para}\n'
                            html += f'                    </p>\n'
                html += '                </div>\n'

        html += '            </section>\n'

    html += """        </div>
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
if "chat_summary" not in st.session_state:
    st.session_state.chat_summary = ""
if "generated_html" not in st.session_state:
    st.session_state.generated_html = None
if "generated_image" not in st.session_state:
    st.session_state.generated_image = None
if "html_filename" not in st.session_state:
    st.session_state.html_filename = None

# 사용자 정보 입력
st.subheader("📋 기본 정보")

# 세션 상태에서 기본값 가져오기
default_name = st.session_state.get('sample_name', '김영희')
default_gender = st.session_state.get('sample_gender', '여자')
default_birth_info = st.session_state.get('sample_birth_info', '양력 1988-08-09 辰時 / 음력 1988-06-27 辰時')

# 성별의 인덱스 계산
gender_options = ["남자", "여자"]
default_gender_index = gender_options.index(default_gender) if default_gender in gender_options else 1

user_name = st.text_input("이름", value=default_name, key="user_name_input")
gender = st.selectbox("성별", gender_options, index=default_gender_index, key="gender_input")
birth_info = st.text_input(
    "생년월일 정보",
    value=default_birth_info,
    help="예시: 양력 1988-08-09 辰時 / 음력 1988-06-27 辰時",
    key="birth_info_input"
)

# 입력된 생년월일 정보 파싱
solar_date = ""
lunar_date = ""
birth_time = ""

if birth_info:
    try:
        # "/" 기준으로 양력/음력 분리
        parts = birth_info.split("/")
        if len(parts) >= 2:
            solar_part = parts[0].strip()
            lunar_part = parts[1].strip()

            # 양력 파싱: "양력 1988-08-09 辰時"
            if "양력" in solar_part:
                solar_info = solar_part.replace("양력", "").strip().split()
                if len(solar_info) >= 1:
                    solar_date = solar_info[0]
                if len(solar_info) >= 2:
                    birth_time = solar_info[1]

            # 음력 파싱: "음력 1988-06-27 辰時"
            if "음력" in lunar_part:
                lunar_info = lunar_part.replace("음력", "").strip().split()
                if len(lunar_info) >= 1:
                    lunar_date = lunar_info[0]
    except Exception as e:
        st.warning(f"생년월일 정보 파싱 중 오류: {e}")

st.markdown("---")

# 샘플 데이터 로드 함수
def load_sample_from_html(html_path: str) -> dict:
    """HTML 파일에서 샘플 데이터를 추출"""
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')

        sample_data = {
            'name': '김영희',
            'gender': '여자',
            'birth_info': '양력 1988-08-09 辰時 / 음력 1988-06-27 辰時',
            'sections': {}
        }

        # 섹션 매핑 (HTML의 섹션 제목 -> 입력창 키)
        section_mapping = {
            '핵심포인트': '핵심포인트(새해신수)',
            '올해의 총운': '올해의총운(새해신수)',
            '일년신수(전반기)': '일년신수(전반기)(토정비결)',
            '일년신수(후반기)': '일년신수(후반기)(토정비결)',
            '올해의 연애운': '올해의연애운(토정비결)',
            '올해의 건강운': '올해의건강운(토정비결)',
            '올해의 직장운': '올해의직장운(토정비결)',
            '올해의 소망운': '올해의소망운(토정비결)',
            '올해의 여행·이사운': '올해의여행이사운(새해신수)',
            '올해의 여행이사운': '올해의여행이사운(새해신수)',  # 가운뎃점 없는 버전도 지원
            '월별운': '월별운(새해신수)',
            '재물운의 특성': '재물운의특성(새해신수)',
            '재물 모으는 법': '재물모으는법(새해신수)',
            '재물 손실 막는 법': '재물손실막는법(새해신수)',
            '재물손실막는법': '재물손실막는법(새해신수)',  # 띄어쓰기 없는 버전도 지원
            '현재의 재물운': '현재의재물운(새해신수)',
            '시기적 운세': '시기적운세(새해신수)',
            '현재의 길흉사': '현재의길흉사(새해신수)',
            '현재의 길흉사운': '현재의길흉사(새해신수)',
            '운명 뛰어넘기': '운명뛰어넘기(새해신수)',
            '운명뛰어넘기': '운명뛰어넘기(새해신수)'  # 띄어쓰기 없는 버전도 지원
        }

        # 모든 섹션 추출
        sections = soup.find_all('section')
        for section in sections:
            h2 = section.find('h2')
            if h2:
                title = h2.get_text(strip=True)
                if title == '그림으로 보는 새해운세':
                    continue

                # 대길대흉 섹션 특별 처리
                if title == '대길대흉':
                    # 대길 추출
                    daegil_div = section.find('div', class_='bg-blue-50')
                    if daegil_div:
                        daegil_parts = []
                        for elem in daegil_div.find_all(['h4', 'p']):
                            if elem.name == 'h4':
                                text = elem.get_text(strip=True)
                                if text:
                                    daegil_parts.append(f"\n{text}\n")
                            elif elem.name == 'p':
                                text = elem.get_text(strip=True)
                                if text:
                                    daegil_parts.append(text)
                        daegil_content = '\n'.join(daegil_parts).strip()
                        if daegil_content:
                            sample_data['sections']['대길(새해신수)'] = daegil_content

                    # 대흉 추출
                    daeheung_div = section.find('div', class_='bg-red-50')
                    if daeheung_div:
                        daeheung_parts = []
                        for elem in daeheung_div.find_all(['h4', 'p']):
                            if elem.name == 'h4':
                                text = elem.get_text(strip=True)
                                if text:
                                    daeheung_parts.append(f"\n{text}\n")
                            elif elem.name == 'p':
                                text = elem.get_text(strip=True)
                                if text:
                                    daeheung_parts.append(text)
                        daeheung_content = '\n'.join(daeheung_parts).strip()
                        if daeheung_content:
                            sample_data['sections']['대흉(새해신수)'] = daeheung_content
                    continue

                # 월별운 특별 처리
                if title == '월별운':
                    month_divs = section.find_all('div', class_='bg-gray-50')
                    month_parts = []
                    for month_div in month_divs:
                        h4 = month_div.find('h4')
                        p = month_div.find('p')
                        if h4 and p:
                            month_title = h4.get_text(strip=True)
                            month_text = p.get_text(strip=True)
                            month_parts.append(f"{month_title}\n{month_text}")
                    if month_parts:
                        sample_data['sections']['월별운(새해신수)'] = '\n'.join(month_parts)
                    continue

                # 일반 섹션 처리
                content_parts = []

                # h3와 p 태그 찾기
                for elem in section.find_all(['h3', 'p']):
                    if elem.name == 'h3':
                        text = elem.get_text(strip=True)
                        if text:
                            content_parts.append(f"\n{text}\n")
                    elif elem.name == 'p':
                        text = elem.get_text(strip=True)
                        if text:
                            content_parts.append(text)

                content = '\n'.join(content_parts).strip()

                # 매핑된 키로 저장
                mapped_key = section_mapping.get(title, title)
                if content:
                    sample_data['sections'][mapped_key] = content

        return sample_data
    except Exception as e:
        st.error(f"샘플 로드 실패: {e}")
        return None

st.subheader("📝 19개 항목 입력")

# 샘플 넣기 버튼
if st.button("📋 샘플 넣기", help="index.html의 내용으로 모든 입력창을 채웁니다"):
    # 현재 스크립트 위치 기준으로 docs/index.html 경로 설정
    sample_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "index.html")

    if not os.path.exists(sample_path):
        st.error(f"⚠️ 샘플 파일을 찾을 수 없습니다: {sample_path}")
        st.info("💡 docs/index.html 파일이 프로젝트에 포함되어 있는지 확인해주세요.")
    else:
        sample_data = load_sample_from_html(sample_path)

        if sample_data:
            # 세션 상태에 샘플 데이터 저장
            st.session_state['sample_loaded'] = True
            st.session_state['sample_name'] = sample_data['name']
            st.session_state['sample_gender'] = sample_data['gender']
            st.session_state['sample_birth_info'] = sample_data['birth_info']
            st.session_state['sample_sections'] = sample_data['sections']
            st.success("✅ 샘플 데이터를 불러왔습니다!")
            st.rerun()

# 샘플 데이터가 로드되었으면 기본 정보는 이미 위의 입력창에서 세션 상태로 반영됨

# 19개 입력창
sections = {}
section_titles = [
    "핵심포인트(새해신수)", "올해의총운(새해신수)", "일년신수(전반기)(토정비결)", "일년신수(후반기)(토정비결)",
    "올해의연애운(토정비결)", "올해의건강운(토정비결)", "올해의직장운(토정비결)", "올해의소망운(토정비결)",
    "올해의여행이사운(새해신수)", "월별운(새해신수)", "재물운의특성(새해신수)", "재물모으는법(새해신수)",
    "재물손실막는법(새해신수)", "현재의재물운(새해신수)", "시기적운세(새해신수)", "대길(새해신수)",
    "대흉(새해신수)", "현재의길흉사(새해신수)", "운명뛰어넘기(새해신수)"
]

for title in section_titles:
    # 샘플 데이터가 있으면 사용
    default_value = ""
    if 'sample_sections' in st.session_state and title in st.session_state['sample_sections']:
        default_value = st.session_state['sample_sections'][title]

    sections[title] = st.text_area(title, value=default_value, height=100, key=title)

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

chat_summary_prompt_input = st.text_area(
    "채팅방 요약 시스템 프롬프트",
    value=DEFAULT_CHAT_SUMMARY_INSTRUCTION,
    height=150,
    help="채팅방 요약 생성 모델에 전달할 시스템 메시지입니다. {user_name}은 자동으로 치환됩니다.",
)
chat_summary_prompt = chat_summary_prompt_input if chat_summary_prompt_input.strip() else DEFAULT_CHAT_SUMMARY_INSTRUCTION

st.markdown("---")

# 두 개의 버튼을 나란히 배치
col1, col2 = st.columns(2)
with col1:
    generate = st.button("🚀 HTML 생성", type="primary", use_container_width=True)
with col2:
    generate_summary = st.button("💬 채팅방 요약", use_container_width=True)

if generate:
    # "올해의총운" 텍스트로 이미지 생성
    base_text = sections.get("올해의총운(새해신수)", "").strip()
    if not base_text:
        st.error("'올해의총운'을 입력해주세요. 이 내용으로 이미지를 생성합니다.")
        st.stop()

    # 이미지 생성 시작 시점의 설정을 고정
    locked_system_prompt = system_prompt
    locked_summary_prompt = summary_prompt
    locked_chat_summary_prompt = chat_summary_prompt
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
                gender=gender,
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

    # 이미지 생성
    with st.spinner("🎨 이미지 생성 중..."):
        imgs = generate_images(
            final_prompt,
            num_images=1,
            provider="openai",
            gemini_client=None,
            openai_client=locked_openai_client,
        )

    # 이미지 처리
    valid = [i for i in imgs if i is not None]
    if not valid:
        st.error("이미지 생성에 실패했습니다.")
        st.stop()

    # 이미지 표시
    st.markdown("#### 🎨 생성된 이미지")
    img = valid[0]
    st.image(img, caption="새해운세 이미지", use_container_width=True)

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

    # HTML 생성 - 섹션 키 매핑 (입력창 키 -> HTML 표시용 키)
    with st.spinner("📄 HTML 생성 중..."):
        # 섹션 키를 HTML 생성 함수가 기대하는 형식으로 변환
        mapped_sections = {}
        for key, content in sections.items():
            # "(새해신수)", "(토정비결)" 등을 제거하여 간단한 키로 변환
            clean_key = key.replace("(새해신수)", "").replace("(토정비결)", "").replace(")", "")
            mapped_sections[clean_key] = content

        html_content = generate_html(
            user_name=user_name,
            gender=gender,
            solar_date=solar_date,
            lunar_date=lunar_date,
            birth_time=birth_time,
            sections=mapped_sections,
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

    st.success(f"✅ HTML 생성 완료!")

# 채팅방 요약 버튼 클릭 시
if generate_summary:
    # 모든 섹션 내용 합치기
    all_content = []
    for title, content in sections.items():
        if content.strip():
            all_content.append(f"## {title}\n{content}")

    full_text = "\n\n".join(all_content)

    if not full_text.strip():
        st.error("입력된 내용이 없습니다. 섹션을 입력해주세요.")
        st.stop()

    # 현재 설정을 고정
    locked_chat_summary_prompt = chat_summary_prompt
    locked_openai_client = openai_client

    with st.spinner("💬 채팅방 요약 생성 중 (gpt-4.1-mini 사용)..."):
        try:
            # 도사 스타일 요약 프롬프트 - {user_name} 치환
            chat_summary_instruction = locked_chat_summary_prompt.format(user_name=user_name)

            chat_summary_msg = f"""다음은 {user_name}의 사주 내용입니다. 이를 도사 말투로 4500자 내외로 요약해주세요:

{full_text}

[요구사항]
- 도사 말투 사용
- {user_name}을(를) 호칭으로 사용
- 핵심 내용 포함: 총운, 연애운, 건강운, 직장운, 재물운, 월별운, 대길대흉 등
- 4500자 내외 (최대 5000자)
- 밝고 유쾌하면서도 무게감 있게"""

            chat_summary = locked_openai_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": chat_summary_instruction},
                    {"role": "user", "content": chat_summary_msg},
                ]
            )
            chat_summary_text = (chat_summary.choices[0].message.content or "").strip()

            # 세션 상태에 채팅방 요약 저장
            st.session_state["chat_summary"] = chat_summary_text

            # 요약 표시
            st.markdown("#### 💬 채팅방 요약")
            if chat_summary_text:
                # 말풍선 UI 스타일로 표시
                st.markdown(f"""
                <div style="display: flex; align-items: flex-start; margin: 20px 0;">
                    <div style="flex-shrink: 0; margin-right: 12px;">
                        <div style="width: 48px; height: 48px; border-radius: 50%; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); display: flex; align-items: center; justify-content: center; font-size: 24px;">
                            🪭
                        </div>
                    </div>
                    <div style="flex-grow: 1; max-width: 85%;">
                        <div style="background-color: #f8f9fa; border-radius: 18px; padding: 16px 20px; position: relative; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                            <div style="font-weight: 600; color: #667eea; margin-bottom: 8px; font-size: 14px;">도사</div>
                            <div style="white-space: pre-wrap; line-height: 1.7; color: #2c3e50; font-size: 15px; max-height: 600px; overflow-y: auto;">
{chat_summary_text}
                            </div>
                            <div style="margin-top: 12px; padding-top: 8px; border-top: 1px solid #e0e0e0; font-size: 11px; color: #999;">
                                📏 {len(chat_summary_text)}자
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.success("✅ 채팅방 요약 생성 완료!")
            else:
                st.warning("채팅방 요약 생성에 실패했습니다.")
        except Exception as exc:
            st.error(f"채팅방 요약 생성 중 오류가 발생했습니다: {exc}")

# 결과물 표시 (세션 상태에서 가져옴)
if st.session_state.generated_html is not None:
    st.markdown("---")
    st.markdown("### 🎨 생성 결과")

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

if not generate and not generate_summary:
    summary_display = st.session_state.get("core_scene_summary", "").strip()
    if summary_display:
        st.markdown("#### ✨ 핵심 장면 요약")
        st.write(summary_display)

    chat_summary_display = st.session_state.get("chat_summary", "").strip()
    if chat_summary_display:
        st.markdown("#### 💬 채팅방 요약 (이전 생성 결과)")
        # 말풍선 UI 스타일로 표시
        st.markdown(f"""
        <div style="display: flex; align-items: flex-start; margin: 20px 0;">
            <div style="flex-shrink: 0; margin-right: 12px;">
                <div style="width: 48px; height: 48px; border-radius: 50%; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); display: flex; align-items: center; justify-content: center; font-size: 24px;">
                    🪭
                </div>
            </div>
            <div style="flex-grow: 1; max-width: 85%;">
                <div style="background-color: #f8f9fa; border-radius: 18px; padding: 16px 20px; position: relative; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                    <div style="font-weight: 600; color: #667eea; margin-bottom: 8px; font-size: 14px;">도사</div>
                    <div style="white-space: pre-wrap; line-height: 1.7; color: #2c3e50; font-size: 15px; max-height: 600px; overflow-y: auto;">
{chat_summary_display}
                    </div>
                    <div style="margin-top: 12px; padding-top: 8px; border-top: 1px solid #e0e0e0; font-size: 11px; color: #999;">
                        📏 {len(chat_summary_display)}자
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
