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
# 로그인 체크 (비활성화)
# ----------------------------
# 로그인 과정 제거됨

# ----------------------------
# 설정
# ----------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TEXT_MODEL = "gemini-3-pro-preview"                 # 프롬프트 작성 모델
IMAGE_MODEL = "gemini-3-pro-image-preview"  # 이미지 생성 모델
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
DEFAULT_BUJEOK_INSTRUCTION = (
    "Create a vertical traditional Korean bujeok talisman artwork in a 9:16 aspect ratio. "
    "The artwork must strongly incorporate visual symbols, objects, patterns, and traditional motifs directly representing {theme_name} and {theme_keywords}. "
    "Use auspicious iconography and lucky cultural elements that are specifically associated with {theme_keywords}, such as emblematic shapes, spiritual objects, charms, or symbolic animals, integrating them into the talisman composition. "
    "Surround the character with detailed brushstroke patterns and ritual symbols that amplify the meaning of {theme_keywords}, visually expressing themes like protection, prosperity, love, success, health, or spiritual blessing depending on the keywords. "
    "Use a 3D sculpted style with soft cinematic lighting, rich depth, elegant shading, and luxurious material texture on aged yellow parchment with weathered ancient Korean paper texture. "
    "Isolated on a clean white background. "
    "No real text, letters, numbers, or watermarks."
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
- 맨 마지막에 더 자세히 보려면 토정비결 보기 버튼을 눌러보라고 안내해"""

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
        # httpx 클라이언트로 프록시 환경 변수 자동 적용 (trust_env=True가 기본값)
        try:
            import httpx
            # trust_env=True로 HTTP_PROXY, HTTPS_PROXY 환경 변수 사용
            http_client = httpx.Client(trust_env=True)
            client = OpenAI(api_key=OPENAI_API_KEY, http_client=http_client)
            return client
        except ImportError:
            # httpx가 없으면 기본 클라이언트 사용
            client = OpenAI(api_key=OPENAI_API_KEY)
            return client
    except Exception as e:
        st.warning(f"OpenAI 클라이언트 초기화 실패: {e}")
        return None


def summarize_to_three_lines(
    source_text: str,
    openai_client: Optional[OpenAI] = None,
) -> str:
    """
    텍스트를 3줄로 요약
    """
    system_instruction = """당신은 사주 내용을 간결하게 요약하는 전문가입니다.

요약 규칙:
- 정확히 3줄로 요약
- 각 줄은 핵심 포인트 하나씩
- 간결하고 명확하게
- 이모지 사용 금지"""

    user_msg = f"""다음 총운 내용을 정확히 3줄로 요약해주세요:

{source_text}

[요구사항]
- 3줄로 요약
- 각 줄은 한 문장
- 핵심 메시지만 전달"""

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
        model="gpt-4.1",
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
            from google.genai import types
            response = gemini_client.models.generate_content(
                model=IMAGE_MODEL,
                contents=f"Create a picture of: {prompt}",
                config=types.GenerateContentConfig(
                    image_config=types.ImageConfig(
                        aspect_ratio="9:16",
                        image_size="4K"
                    )
                )
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

def generate_bujeok_image_single(prompt: str, image_path: str, openai_client: OpenAI):
    """프롬프트로 단일 부적 이미지를 생성하는 함수 (병렬 처리용)"""
    # images.edit 사용하여 캐릭터 보존하면서 스타일 변경
    with open(image_path, "rb") as img_file:
        response = openai_client.images.edit(
            model="gpt-image-1",
            image=img_file,
            prompt=prompt,
            n=1,
            size="1024x1536"
        )
    
    if response.data:
        img_data = response.data[0]
        if getattr(img_data, "url", None):
            image_bytes = requests.get(img_data.url).content
        else:
            image_bytes = base64.b64decode(img_data.b64_json)
        
        img = Image.open(BytesIO(image_bytes)).convert("RGBA")
        return img
    return None

def generate_bujeok_images(base_prompt: str, char_images: list, openai_client: OpenAI):
    """
    여러 캐릭터 이미지로 부적 이미지들을 병렬로 생성
    char_images: [(name, path), ...] 형식의 리스트
    반환: [(name, prompt, image), ...] 형식의 리스트
    """
    results = []
    images = [None] * len(char_images)
    
    # base_prompt를 직접 사용하여 모든 이미지를 동시에 생성
    with ThreadPoolExecutor(max_workers=len(char_images)) as executor:
        future_to_index = {}
        for i, (char_name, img_path) in enumerate(char_images):
            future = executor.submit(generate_bujeok_image_single, base_prompt, img_path, openai_client)
            future_to_index[future] = i
        
        # 완료된 이미지들 수집
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                images[index] = future.result()
            except Exception as exc:
                # 에러 메시지는 streamlit 밖에서 발생하므로 무시
                images[index] = None
    
    # 결과 조합
    for i, (char_name, _) in enumerate(char_images):
        results.append((char_name, base_prompt, images[i]))
    
    return results

def generate_html(user_name: str, gender: str, solar_date: str, lunar_date: str,
                  birth_time: str, sections: dict, image_base64: str,
                  chongun_summary: str = "", bujeok_images: list = None) -> str:
    """
    19개 섹션 내용을 받아서 HTML을 생성
    image_base64: base64로 인코딩된 이미지 데이터
    chongun_summary: 총운 3줄 요약
    bujeok_images: 부적 이미지 리스트 [(char_name, theme_name, model_name, base64), ...]
    """
    # 디버깅: HTML 생성 함수에 전달된 sections 확인 (주석 처리 - 필요시 활성화)
    # import sys
    # print(f"[HTML DEBUG] generate_html 함수 시작", file=sys.stderr)
    # print(f"[HTML DEBUG] sections 키 목록: {list(sections.keys())}", file=sys.stderr)
    
    if bujeok_images is None:
        bujeok_images = []
    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{user_name} 님의 신년운세</title>
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
        /* Sticky 헤더가 메인 카드의 border-radius를 넘어가지 않도록 */
        .sticky-header {{
            position: sticky;
            top: 0;
            z-index: 50;
            background-color: white;
            border-bottom: 1px solid #e5e7eb;
        }}
        /* 스크롤 시 sticky 헤더 아래 여백 */
        html {{
            scroll-padding-top: 210px;
        }}
        /* 앵커 포인트 스타일 - 타이틀 1픽셀 상단 */
        .anchor-point {{
            display: block;
            position: relative;
            top: -1px;
            visibility: hidden;
        }}
    </style>
</head>
<body class="bg-gray-100 py-10 px-4">

    <!-- 메인 콘텐츠 카드 -->
    <main class="max-w-3xl mx-auto bg-white shadow-2xl rounded-xl">
        <!-- 고정 헤더 영역 -->
        <div class="sticky-header rounded-t-xl">
            <div class="p-6 sm:p-8 pb-0">
                <!-- 제목 -->
                <h1 class="text-3xl sm:text-4xl font-bold text-gray-800 mb-4 text-center">
                    {user_name} 님의 신년운세
                </h1>

                <!-- 네비게이션 버튼 (가로 스크롤) -->
                <div class="-mx-8 px-8">
                    <div class="overflow-x-auto pb-2">
                        <div class="flex gap-3 min-w-max">
                            <a href="#section-총운" class="px-4 py-2 bg-blue-100 text-blue-700 rounded-full font-medium hover:bg-blue-200 transition whitespace-nowrap">총운</a>
                            <a href="#section-기운흐름" class="px-4 py-2 bg-indigo-100 text-indigo-700 rounded-full font-medium hover:bg-indigo-200 transition whitespace-nowrap">기운흐름</a>
                            <a href="#section-테마-운세" class="px-4 py-2 bg-purple-100 text-purple-700 rounded-full font-medium hover:bg-purple-200 transition whitespace-nowrap">테마 운세</a>
                            <a href="#section-월별운세" class="px-4 py-2 bg-orange-100 text-orange-700 rounded-full font-medium hover:bg-orange-200 transition whitespace-nowrap">월별운세</a>
                            <a href="#section-운의-흐름" class="px-4 py-2 bg-red-100 text-red-700 rounded-full font-medium hover:bg-red-200 transition whitespace-nowrap">운의 흐름</a>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- 콘텐츠 영역 -->
        <div class="p-8 sm:p-12 pt-2">
"""

    # 섹션을 묶어서 처리 (새로운 순서)
    grouped_sections = [
        {
            "title": "총운",
            "sections": ["핵심포인트", "올해의총운"],
            "color": "blue"
        },
        {
            "title": "그림으로 보는 새해운세",
            "sections": ["__image__"],  # 특별 처리: 이미지
            "color": "blue"
        },
        {
            "title": "기운흐름",
            "sections": ["일년신수(전반기", "일년신수(후반기"],
            "color": "indigo"
        },
        {
            "title": "테마 운세",
            "sections": ["재물모으는법", "현재의재물운", 
                        "올해의연애운", "올해의건강운", "올해의직장운", "올해의소망운", "올해의여행이사운"],
            "color": "purple"
        },
        {
            "title": "월별운세",
            "sections": ["월별운"],
            "color": "orange"
        },
        {
            "title": "운의 흐름",
            "sections": ["시기적운세", "대길", "대흉", "현재의길흉사", "운명뛰어넘기"],
            "color": "red"
        }
    ]

    for group in grouped_sections:
        display_title = group["title"]
        section_keys = group["sections"]
        color = group["color"]

        # 이미지 섹션 특별 처리
        if section_keys == ["__image__"]:
            section_id = display_title.replace(" ", "-")
            html += f"""
            <!-- 앵커 포인트 -->
            <span id="section-{section_id}" class="anchor-point"></span>

            <!-- 섹션: 그림 -->
            <section class="mb-10">
                <p class="text-center text-lg font-medium text-gray-600 mb-6">
                    이미지로 보는 내 사주
                </p>
"""
            # 총운 3줄 요약을 이미지 위에 표시
            if chongun_summary:
                html += f"""
                <!-- 총운 3줄 요약 -->
                <div class="mb-6 p-5 bg-blue-50 border-l-4 border-blue-500 rounded-r-lg max-w-2xl mx-auto">
                    <div class="text-base text-gray-800 leading-relaxed whitespace-pre-line">
{chongun_summary}
                    </div>
                </div>
"""
            html += f"""
                <div class="flex justify-center">
                    <img src="data:image/png;base64,{image_base64}" alt="새해운세 이미지" class="rounded-lg shadow-lg max-w-full h-auto">
                </div>
            </section>
"""
            continue

        # 그룹 내 모든 섹션의 내용을 수집
        combined_content = []
        has_content = False

        # 월별운세 디버깅: 실제 HTML에 주석으로 출력
        if display_title == "월별운세":
            html += f"<!-- 월별운세 디버깅: section_keys={section_keys}, sections 키={list(sections.keys())[:5]} -->\n"
            for sk in section_keys:
                html += f"<!-- 찾는 키: '{sk}', 존재?: {sk in sections}, 내용: {len(sections.get(sk, ''))}자 -->\n"

        for key in section_keys:
            content = sections.get(key, "").strip()
            if content:
                has_content = True
                combined_content.append(content)

        # 내용이 없으면 건너뛰기
        if not has_content:
            if display_title == "월별운세":
                import sys
                print(f"[DEBUG] 월별운세 섹션이 건너뛰어졌습니다. has_content=False", file=sys.stderr)
            continue

        # 섹션 ID 생성 (한글 제목을 그대로 사용)
        section_id = display_title.replace(" ", "-")

        html += f"""
            <!-- 앵커 포인트 -->
            <span id="section-{section_id}" class="anchor-point"></span>

            <!-- 섹션: {display_title} -->
            <section class="mb-10">
                <h2 class="text-2xl font-semibold text-{color}-700 border-b-2 border-{color}-100 pb-3 mb-6">
                    {display_title}
                </h2>
                """

        # 총운 섹션: 서브 타이틀 추가
        if display_title == "총운":
            html += """
                <p class="text-lg font-medium text-gray-600 mb-6">
                    올해의 주제와 흐름
                </p>
"""

        # 월별운세는 특별 처리 (그리드 레이아웃)
        if display_title == "월별운세":
            # 월별 정보 파싱
            content = combined_content[0] if combined_content else ""
            months = []
            lines = content.split('\n')
            current_month = None
            current_text = []

            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # "01월", "1월", "1월 운세" 등의 패턴 찾기
                # '월'이 포함되고 '운세'로 끝나거나, 짧은 월 표기일 경우
                if ('월' in line and '운세' in line) or (line.endswith('월') and len(line) <= 4):
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
        # 테마 운세 섹션 특별 처리
        elif display_title == "테마 운세":
            # 각 운세별로 서브타이틀과 함께 표시
            theme_groups = [
                {
                    "title": "재물운",
                    "keys": ["재물모으는법", "현재의재물운"]
                },
                {
                    "title": "연애운",
                    "keys": ["올해의연애운"]
                },
                {
                    "title": "건강운",
                    "keys": ["올해의건강운"]
                },
                {
                    "title": "직장운",
                    "keys": ["올해의직장운"]
                },
                {
                    "title": "소망운",
                    "keys": ["올해의소망운"]
                },
                {
                    "title": "이사운",
                    "keys": ["올해의여행이사운"]
                }
            ]
            
            for theme in theme_groups:
                theme_title = theme["title"]
                theme_keys = theme["keys"]
                
                # 해당 테마의 내용 수집
                theme_content = []
                for key in theme_keys:
                    content = sections.get(key, "").strip()
                    if content:
                        theme_content.append(content)
                
                # 내용이 있으면 서브타이틀과 함께 표시
                if theme_content:
                    html += f'                <!-- {theme_title} -->\n'
                    html += f'                <div class="mb-8">\n'
                    html += f'                    <h3 class="text-xl font-semibold text-purple-600 mb-4">\n'
                    html += f'                        {theme_title}\n'
                    html += f'                    </h3>\n'
                    
                    # 내용 합치기
                    full_theme_content = '\n\n'.join(theme_content)
                    paragraphs = [p.strip() for p in full_theme_content.split('\n\n') if p.strip()]
                    
                    # 내용 포맷팅
                    formatted_blocks = []
                    for para in paragraphs:
                        lines = [l.strip() for l in para.split('\n') if l.strip()]
                        if len(lines) == 0:
                            continue
                        if len(lines) > 1 and len(lines[0]) < 100:
                            formatted_blocks.append({'type': 'titled', 'title': lines[0], 'paragraphs': lines[1:]})
                        else:
                            formatted_blocks.append({'type': 'plain', 'paragraphs': lines})
                    
                    if formatted_blocks:
                        html += '                    <div class="space-y-4">\n'
                        for block in formatted_blocks:
                            if block['type'] == 'titled':
                                html += '                        <div>\n'
                                html += f'                            <h4 class="text-lg font-semibold text-gray-700 mb-2">{block["title"]}</h4>\n'
                                for i, para in enumerate(block['paragraphs']):
                                    if i == 0:
                                        html += f'                            <p class="text-base text-gray-700 leading-relaxed">{para}</p>\n'
                                    else:
                                        html += f'                            <p class="text-base text-gray-700 leading-relaxed mt-3">{para}</p>\n'
                                html += '                        </div>\n'
                            else:
                                for para in block['paragraphs']:
                                    html += f'                        <p class="text-base text-gray-700 leading-relaxed">{para}</p>\n'
                        html += '                    </div>\n'
                    
                    html += '                </div>\n'
        # 운의 흐름 섹션 특별 처리 (시기적운세, 대길, 대흉, 현재의길흉사, 운명뛰어넘기 포함)
        elif display_title == "운의 흐름":
            # 먼저 시기적운세 표시
            sikijuk_content = sections.get("시기적운세", "").strip()
            if sikijuk_content:
                paragraphs = [p.strip() for p in sikijuk_content.split('\n\n') if p.strip()]
                formatted_blocks = []
                for para in paragraphs:
                    lines = [l.strip() for l in para.split('\n') if l.strip()]
                    if len(lines) == 0:
                        continue
                    if len(lines) > 1 and len(lines[0]) < 100:
                        formatted_blocks.append({'type': 'titled', 'title': lines[0], 'paragraphs': lines[1:]})
                    else:
                        formatted_blocks.append({'type': 'plain', 'paragraphs': lines})

                if formatted_blocks:
                    html += '                <div class="space-y-6 mb-8">\n'
                    for block in formatted_blocks:
                        if block['type'] == 'titled':
                            html += '                    <div>\n'
                            html += f'                        <h3 class="text-xl font-bold text-gray-700 mb-2">{block["title"]}</h3>\n'
                            for i, para in enumerate(block['paragraphs']):
                                if i == 0:
                                    html += f'                        <p class="text-base text-gray-700 leading-relaxed">{para}</p>\n'
                                else:
                                    html += f'                        <p class="text-base text-gray-700 leading-relaxed mt-4">{para}</p>\n'
                            html += '                    </div>\n'
                        else:
                            for para in block['paragraphs']:
                                html += f'                    <p class="text-base text-gray-700 leading-relaxed">{para}</p>\n'
                    html += '                </div>\n'

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

            # 현재의 길흉사 표시
            gilheungsa_content = sections.get("현재의길흉사", "").strip()
            if gilheungsa_content:
                paragraphs = [p.strip() for p in gilheungsa_content.split('\n\n') if p.strip()]
                formatted_blocks = []
                for para in paragraphs:
                    lines = [l.strip() for l in para.split('\n') if l.strip()]
                    if len(lines) == 0:
                        continue
                    if len(lines) > 1 and len(lines[0]) < 100:
                        formatted_blocks.append({'type': 'titled', 'title': lines[0], 'paragraphs': lines[1:]})
                    else:
                        formatted_blocks.append({'type': 'plain', 'paragraphs': lines})

                if formatted_blocks:
                    html += '                <div class="space-y-6 mt-8">\n'
                    for block in formatted_blocks:
                        if block['type'] == 'titled':
                            html += '                    <div>\n'
                            html += f'                        <h3 class="text-xl font-bold text-gray-700 mb-2">{block["title"]}</h3>\n'
                            for i, para in enumerate(block['paragraphs']):
                                if i == 0:
                                    html += f'                        <p class="text-base text-gray-700 leading-relaxed">{para}</p>\n'
                                else:
                                    html += f'                        <p class="text-base text-gray-700 leading-relaxed mt-4">{para}</p>\n'
                            html += '                    </div>\n'
                        else:
                            for para in block['paragraphs']:
                                html += f'                    <p class="text-base text-gray-700 leading-relaxed">{para}</p>\n'
                    html += '                </div>\n'

            # 운명 뛰어넘기 - 서브타이틀로 표시
            unmyung_content = sections.get("운명뛰어넘기", "").strip()
            if unmyung_content:
                html += '                <!-- 운명 뛰어넘기 -->\n'
                html += '                <div class="mt-8">\n'
                html += '                    <h3 class="text-xl font-semibold text-red-600 mb-4">\n'
                html += '                        운명 뛰어넘기\n'
                html += '                    </h3>\n'
                
                paragraphs = [p.strip() for p in unmyung_content.split('\n\n') if p.strip()]
                formatted_blocks = []
                for para in paragraphs:
                    lines = [l.strip() for l in para.split('\n') if l.strip()]
                    if len(lines) == 0:
                        continue
                    if len(lines) > 1 and len(lines[0]) < 100:
                        formatted_blocks.append({'type': 'titled', 'title': lines[0], 'paragraphs': lines[1:]})
                    else:
                        formatted_blocks.append({'type': 'plain', 'paragraphs': lines})

                if formatted_blocks:
                    html += '                    <div class="space-y-4">\n'
                    for block in formatted_blocks:
                        if block['type'] == 'titled':
                            html += '                        <div>\n'
                            html += f'                            <h4 class="text-lg font-semibold text-gray-700 mb-2">{block["title"]}</h4>\n'
                            for i, para in enumerate(block['paragraphs']):
                                if i == 0:
                                    html += f'                            <p class="text-base text-gray-700 leading-relaxed">{para}</p>\n'
                                else:
                                    html += f'                            <p class="text-base text-gray-700 leading-relaxed mt-3">{para}</p>\n'
                            html += '                        </div>\n'
                        else:
                            for para in block['paragraphs']:
                                html += f'                        <p class="text-base text-gray-700 leading-relaxed">{para}</p>\n'
                    html += '                    </div>\n'
                
                html += '                </div>\n'
        else:
            # 일반 섹션 처리 - 여러 섹션의 내용을 합쳐서 표시
            # 합친 내용을 하나의 문자열로 결합
            full_content = '\n\n'.join(combined_content)
            
            # 기운흐름 섹션: 제목 치환
            if display_title == "기운흐름":
                full_content = full_content.replace("이 사주 일년신수 (전반기) 총평", "전반기 기운의 변화")
                full_content = full_content.replace("이 사주 일년신수 (후반기) 총평", "후반기 기운의 변화")

            # 내용을 줄바꿈으로 분리
            paragraphs = [p.strip() for p in full_content.split('\n\n') if p.strip()]

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
                            # 짧은 텍스트(100자 미만)는 볼드 처리
                            if len(para) < 100 and not para.endswith('.') and not para.endswith('다'):
                                html += f'                    <p class="font-bold text-gray-800">\n'
                            else:
                                html += f'                    <p>\n'
                            html += f'                        {para}\n'
                            html += f'                    </p>\n'
                html += '                </div>\n'

        html += '            </section>\n'

    # 부적 이미지 섹션 추가 (맨 마지막)
    if bujeok_images:
        html += """
            <!-- 부적 섹션 -->
            <section class="mb-10 mt-12">
                <div class="text-center">
                    <h2 class="text-2xl font-semibold text-gray-800 mb-6">
                        행운의 부적
                    </h2>
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-8 mt-8">
"""
        for char_name, theme_name, model_name, img_base64 in bujeok_images:
            html += f"""
                        <div class="flex flex-col items-center">
                            <h3 class="text-xl font-semibold text-gray-800 mb-2">{theme_name} 부적</h3>
                            <p class="text-sm text-gray-600 mb-4">{char_name} · {model_name}</p>
                            <img src="data:image/png;base64,{img_base64}" alt="{theme_name} 부적" class="rounded-lg shadow-xl" style="max-height: 600px; width: auto;">
                        </div>
"""
        html += """                    </div>
                </div>
            </section>
"""

    html += """        </div>
    </main>

    <script>
        // 앵커 링크 클릭 시 스크롤만 처리 (페이지 리로드 방지)
        document.addEventListener('DOMContentLoaded', function() {
            // 모든 앵커 링크에 이벤트 리스너 추가
            document.querySelectorAll('a[href^="#"]').forEach(anchor => {
                anchor.addEventListener('click', function(e) {
                    e.preventDefault(); // 기본 동작 방지

                    const targetId = this.getAttribute('href').substring(1);
                    const targetElement = document.getElementById(targetId);

                    if (targetElement) {
                        // 부드러운 스크롤
                        targetElement.scrollIntoView({
                            behavior: 'smooth',
                            block: 'start'
                        });
                    }
                });
            });
        });
    </script>

</body>
</html>
"""
    return html

# ----------------------------
# UI
# ----------------------------
st.title("🧧 신년운세 HTML 생성기")
st.caption("17개 항목을 입력하면 이미지와 함께 HTML을 생성합니다")

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

# CSV 파일 업로드로 샘플 데이터 입력 (위젯 생성 전에 처리)
st.markdown("**📤 샘플 데이터 업로드**")
uploaded_csv = st.file_uploader(
    "CSV 파일을 업로드하면 자동으로 입력창이 채워집니다",
    type=['csv'],
    help="이름, 성별, 생년월일 정보와 19개 섹션 데이터가 포함된 CSV 파일을 업로드하세요"
)

if uploaded_csv is not None:
    # 무한 루프 방지: 파일 이름으로 이미 처리했는지 확인
    csv_file_id = f"{uploaded_csv.name}_{uploaded_csv.size}"

    if st.session_state.get('last_processed_csv') != csv_file_id:
        try:
            import pandas as pd
            import io

            # CSV 파일 읽기
            df = pd.read_csv(io.StringIO(uploaded_csv.getvalue().decode('utf-8')))

            # 필수 컬럼 확인
            required_cols = ['항목', '내용']
            if not all(col in df.columns for col in required_cols):
                st.error(f"⚠️ CSV 파일에 필수 컬럼이 없습니다: {required_cols}")
            else:
                # 데이터 추출
                sample_data = {'sections': {}}

                for _, row in df.iterrows():
                    item = str(row['항목']).strip()
                    content = str(row['내용']).strip()

                    if item == '이름':
                        sample_data['name'] = content
                    elif item == '성별':
                        sample_data['gender'] = content
                    elif item == '생년월일':
                        sample_data['birth_info'] = content
                    else:
                        # 섹션 데이터
                        sample_data['sections'][item] = content

                # 세션 상태에 저장 (위젯 key에 맞춰서)
                if 'name' in sample_data:
                    st.session_state['user_name_input'] = sample_data['name']
                if 'gender' in sample_data:
                    st.session_state['gender_input'] = sample_data['gender']
                if 'birth_info' in sample_data:
                    st.session_state['birth_info_input'] = sample_data['birth_info']
                if sample_data.get('sections'):
                    # 각 섹션의 text_area key에 직접 값 설정
                    loaded_sections = []
                    for section_key, section_value in sample_data['sections'].items():
                        st.session_state[section_key] = section_value
                        loaded_sections.append(section_key)

                # 처리 완료 표시
                st.session_state['last_processed_csv'] = csv_file_id
                st.session_state['loaded_sections_debug'] = loaded_sections

                st.success(f"✅ CSV 파일에서 데이터를 불러왔습니다! (이름: {sample_data.get('name')}, 섹션: {len(sample_data.get('sections', {}))}개)")
                with st.expander("🔍 로드된 섹션 키 확인"):
                    for key in loaded_sections[:5]:
                        st.write(f"• {key}")
                st.rerun()

        except Exception as e:
            st.error(f"⚠️ CSV 파일 읽기 실패: {e}")
            st.info("💡 CSV 파일 형식을 확인해주세요. 첫 행은 '항목,내용' 헤더여야 합니다.")
    else:
        st.info("✅ CSV 데이터가 이미 로드되었습니다. 아래 입력창에서 확인하세요.")

st.markdown("---")

# 사용자 정보 입력
st.subheader("📋 기본 정보")

# 세션 상태 초기값 설정 (최초 실행 시에만)
if 'user_name_input' not in st.session_state:
    st.session_state['user_name_input'] = '김영희'
if 'gender_input' not in st.session_state:
    st.session_state['gender_input'] = '여자'
if 'birth_info_input' not in st.session_state:
    st.session_state['birth_info_input'] = '양력 1988-08-09 辰時 / 음력 1988-06-27 辰時'

# 성별 옵션
gender_options = ["남자", "여자"]

# 위젯 (key로 세션 상태가 자동 연결됨)
user_name = st.text_input("이름", key="user_name_input")
gender = st.selectbox("성별", gender_options, key="gender_input")
birth_info = st.text_input(
    "생년월일 정보",
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

# 내장 샘플 데이터 (Render 배포 시 파일 의존성 제거)
EMBEDDED_SAMPLE_DATA = None  # 초기화는 함수에서 수행

def get_embedded_sample_data() -> dict:
    """내장된 샘플 데이터 반환 (JSON 파일에서 로드)"""
    global EMBEDDED_SAMPLE_DATA
    if EMBEDDED_SAMPLE_DATA is not None:
        return EMBEDDED_SAMPLE_DATA

    # JSON 파일이 있으면 로드 (여러 경로 시도)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(current_dir, "extracted_sample_data.json"),
        "extracted_sample_data.json",  # 현재 작업 디렉토리
        os.path.join(os.getcwd(), "extracted_sample_data.json")
    ]

    json_path = None
    for path in possible_paths:
        if os.path.exists(path):
            json_path = path
            break

    if json_path and os.path.exists(json_path):
        try:
            import json
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 섹션 키 매핑 (HTML 섹션명 -> 입력창 키)
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
                '월별운': '월별운(새해신수)',
                '재물 모으는 법': '재물모으는법(새해신수)',
                '현재의 재물운': '현재의재물운(새해신수)',
                '시기적 운세': '시기적운세(새해신수)',
                '현재의 길흉사': '현재의길흉사(새해신수)',
                '운명 뛰어넘기': '운명뛰어넘기(새해신수)'
            }

            # 매핑된 섹션 생성
            mapped_sections = {}
            for old_key, content in data['sections'].items():
                if old_key == '그림으로 보는 새해운세':
                    continue  # 이미지는 제외
                new_key = section_mapping.get(old_key, old_key)
                mapped_sections[new_key] = content

            # 대길대흉 섹션 분리
            if '대길대흉' in data['sections']:
                daegil_daeheung = data['sections']['대길대흉']
                # 간단한 분리: "대흉" 키워드로 나누기
                if '대흉 (大凶)' in daegil_daeheung:
                    parts = daegil_daeheung.split('대흉 (大凶)')
                    mapped_sections['대길(새해신수)'] = parts[0].replace('대길 (大吉)', '').strip()
                    mapped_sections['대흉(새해신수)'] = parts[1].strip()
                else:
                    mapped_sections['대길(새해신수)'] = daegil_daeheung
                    mapped_sections['대흉(새해신수)'] = ""

            EMBEDDED_SAMPLE_DATA = {
                'name': data['name'],
                'gender': data['gender'],
                'birth_info': data['birth_info'],
                'sections': mapped_sections
            }
            return EMBEDDED_SAMPLE_DATA
        except Exception as e:
            st.warning(f"JSON 샘플 데이터 로드 실패: {e}")
    else:
        # 디버깅: 파일을 찾을 수 없을 때 경로 정보 출력
        st.warning(f"⚠️ extracted_sample_data.json 파일을 찾을 수 없습니다.")
        st.info(f"시도한 경로:\n" + "\n".join(f"- {p} (존재: {os.path.exists(p)})" for p in possible_paths))

    # JSON 파일이 없으면 기본 샘플 데이터 반환
    return {
        'name': '김영희',
        'gender': '여자',
        'birth_info': '양력 1988-08-09 辰時 / 음력 1988-06-27 辰時',
        'sections': {}
    }

# 샘플 데이터 로드 함수
def load_sample_from_html(html_path: str) -> dict:
    """HTML 파일에서 샘플 데이터를 추출 (파일이 없으면 내장 데이터 사용)"""
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        from bs4 import BeautifulSoup
        import re
        soup = BeautifulSoup(html_content, 'html.parser')

        sample_data = {
            'name': '김영희',
            'gender': '여자',
            'birth_info': '양력 1988-08-09 辰時 / 음력 1988-06-27 辰時',
            'sections': {}
        }

        # HTML에서 기본정보 추출
        # 1. 제목에서 이름 추출 (예: "김영희 님의 토정비결")
        h1 = soup.find('h1')
        if h1:
            title_text = h1.get_text(strip=True)
            name_match = re.search(r'(.+?)\s*님의', title_text)
            if name_match:
                sample_data['name'] = name_match.group(1).strip()

        # 2. 사용자 정보에서 성별과 생년월일 추출
        # 예: "[ 여자 ] 양력 1988-08-09 辰時 / 음력 1988-06-27 辰時"
        user_info_p = soup.find('p', class_='text-lg')
        if user_info_p:
            info_text = user_info_p.get_text(strip=True)

            # 성별 추출
            gender_match = re.search(r'\[\s*(남자|여자)\s*\]', info_text)
            if gender_match:
                sample_data['gender'] = gender_match.group(1)

            # 생년월일 정보 추출 ([ 성별 ] 이후의 모든 텍스트)
            birth_match = re.search(r'\]\s*(.+)', info_text)
            if birth_match:
                sample_data['birth_info'] = birth_match.group(1).strip()

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
            '재물 모으는 법': '재물모으는법(새해신수)',
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

# 섹션 제목 정의
section_titles = [
    "핵심포인트(새해신수)", "올해의총운(새해신수)", "일년신수(전반기)(토정비결)", "일년신수(후반기)(토정비결)",
    "올해의연애운(토정비결)", "올해의건강운(토정비결)", "올해의직장운(토정비결)", "올해의소망운(토정비결)",
    "올해의여행이사운(새해신수)", "월별운(새해신수)", "재물모으는법(새해신수)",
    "현재의재물운(새해신수)", "시기적운세(새해신수)", "대길(새해신수)",
    "대흉(새해신수)", "현재의길흉사(새해신수)", "운명뛰어넘기(새해신수)"
]

# 디버깅: 세션 상태 확인
debug_sections = [key for key in section_titles if key in st.session_state and st.session_state[key]]
if debug_sections:
    st.info(f"🔍 세션 상태에 데이터가 있는 섹션: {len(debug_sections)}개")
    with st.expander("세션 상태 디버그 정보"):
        for key in debug_sections[:5]:
            st.write(f"• {key}: {len(st.session_state[key])} 문자")

# CSV 로드 디버깅
if 'loaded_sections_debug' in st.session_state:
    loaded = st.session_state['loaded_sections_debug']
    st.info(f"📥 CSV에서 로드된 섹션: {len(loaded)}개")
    with st.expander("CSV 로드 디버그 정보"):
        st.write("CSV에서 로드된 키:")
        for key in loaded[:5]:
            st.write(f"• {key}")
        st.write("\n코드에서 기대하는 키 (처음 5개):")
        for key in section_titles[:5]:
            st.write(f"• {key}")
        st.write("\n세션 상태 실제 값 샘플:")
        for key in loaded[:2]:
            if key in st.session_state:
                st.write(f"✅ {key}: {st.session_state[key][:50]}..." if len(st.session_state[key]) > 50 else f"✅ {key}: {st.session_state[key]}")
            else:
                st.write(f"❌ {key}: 세션 상태에 없음")

# 19개 입력창
sections = {}

for title in section_titles:
    # 세션 상태의 값을 가져와서 value로 전달 (경고 발생하지만 작동함)
    # key를 함께 사용하여 변경사항이 세션 상태에 저장됨
    default_value = st.session_state.get(title, "")
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

bujeok_prompt_input = st.text_area(
    "부적 이미지 시스템 프롬프트",
    value=DEFAULT_BUJEOK_INSTRUCTION,
    height=120,
    help="부적 이미지 생성 시 사용할 시스템 프롬프트입니다. {theme_name}과 {theme_keywords}는 자동으로 치환됩니다.",
)
bujeok_prompt = bujeok_prompt_input if bujeok_prompt_input.strip() else DEFAULT_BUJEOK_INSTRUCTION

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
    # 시작 시간 기록
    start_time = time.time()

    # "올해의총운" 텍스트로 이미지 생성
    base_text = sections.get("올해의총운(새해신수)", "").strip()
    if not base_text:
        st.error("'올해의총운'을 입력해주세요. 이 내용으로 이미지를 생성합니다.")
        st.stop()

    # 이미지 생성 시작 시점의 설정을 고정
    locked_system_prompt = system_prompt
    locked_summary_prompt = summary_prompt
    locked_bujeok_prompt = bujeok_prompt
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

    # 총운 3줄 요약 생성
    with st.spinner("📋 총운 요약 생성 중 (gpt-4.1-mini 사용)..."):
        try:
            chongun_text = sections.get("핵심포인트(새해신수)", "").strip() + "\n\n" + sections.get("올해의총운(새해신수)", "").strip()
            chongun_summary = summarize_to_three_lines(
                chongun_text,
                openai_client=locked_openai_client
            )
        except Exception as exc:
            st.warning(f"총운 요약 생성 중 오류: {exc}")
            chongun_summary = ""

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
    timestamp = int(time.time())

    # 사주 이미지 생성 함수
    def generate_saju_image():
        try:
            imgs = generate_images(
                final_prompt,
                num_images=1,
                provider="openai",
                gemini_client=None,
                openai_client=locked_openai_client,
            )
            valid = [i for i in imgs if i is not None]
            return {"success": True, "image": valid[0] if valid else None, "error": None}
        except Exception as e:
            return {"success": False, "image": None, "error": str(e)}

    # 부적 이미지 생성 함수 (OpenAI와 Gemini 각각 1개씩)
    def generate_bujeok_images_wrapper():
        try:
            import random
            img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img")
            char_images = [
                ("나나", os.path.join(img_dir, "nana.png")),
                ("뱐냐", os.path.join(img_dir, "Bbanya.png")),
                ("앙몬드", os.path.join(img_dir, "Angmond.png"))
            ]
            
            valid_chars = [(name, path) for name, path in char_images if os.path.exists(path)]
            
            if valid_chars and (locked_openai_client or gemini_client):
                # 랜덤으로 캐릭터 2개 선택 (OpenAI용, Gemini용)
                selected_chars = random.sample(valid_chars, min(2, len(valid_chars)))
                if len(selected_chars) == 1:
                    selected_chars = [selected_chars[0], selected_chars[0]]  # 캐릭터가 1개뿐이면 중복 사용
                
                # 랜덤으로 테마 2개 선택
                themes = [
                    {"name": "재물운", "keywords": "wealth, prosperity, fortune, gold coins, money"},
                    {"name": "연애운", "keywords": "love, romance, heart, relationships, harmony"},
                    {"name": "건강운", "keywords": "health, vitality, wellness, energy, longevity"},
                    {"name": "직장운", "keywords": "career, success, achievement, growth, promotion"},
                    {"name": "소망운", "keywords": "wishes, dreams, goals, aspirations, fulfillment"},
                    {"name": "이사운", "keywords": "moving, new home, journey, change, fresh start"}
                ]
                selected_themes = random.sample(themes, min(2, len(themes)))
                if len(selected_themes) == 1:
                    selected_themes = [selected_themes[0], selected_themes[0]]
                
                enhanced_results = []
                
                # OpenAI로 부적 생성 (캐릭터 부적 - 이미지 편집)
                if locked_openai_client:
                    openai_prompt = locked_bujeok_prompt.format(
                        theme_name=selected_themes[0]['name'],
                        theme_keywords=selected_themes[0]['keywords']
                    )
                    openai_results = generate_bujeok_images(openai_prompt, [selected_chars[0]], locked_openai_client)
                    if openai_results and openai_results[0][2] is not None:
                        enhanced_results.append((
                            openai_results[0][0],  # 캐릭터 이름
                            selected_themes[0]['name'], 
                            "OpenAI (캐릭터 부적)",
                            openai_results[0][1], 
                            openai_results[0][2]
                        ))
                
                # Gemini로 부적 생성 (캐릭터 부적 - multimodal 입력 사용)
                if gemini_client:
                    try:
                        # 캐릭터 이미지 로드
                        char_name, char_path = selected_chars[1]
                        char_image = Image.open(char_path).convert("RGBA")
                        
                        # 1단계: gemini-3-pro-preview로 캐릭터 초상세 분석
                        analysis_prompt = """Analyze this character image in EXTREME DETAIL for image generation. Provide:

1. EXACT Physical Appearance:
   - Face: Eye shape, size, color, expression, eyebrow style, nose shape, mouth shape, skin tone
   - Hair: Exact style, length, color, texture, accessories
   - Body: Build, height proportions, pose, gesture
   - Every visible detail

2. EXACT Clothing & Accessories:
   - Every piece of clothing with colors, patterns, textures
   - All accessories, jewelry, props with exact descriptions
   - Material appearance (fabric, metal, etc.)

3. Art Style & Rendering:
   - Specific style name (3D, anime, cartoon, etc.)
   - Line work, shading technique, rendering quality
   - Texture and material details

4. Color Palette:
   - Dominant colors with specific shades
   - Lighting direction and color temperature
   - Shadow and highlight colors

5. Unique Identifying Features:
   - Any distinctive marks, expressions, or characteristics
   - Character personality conveyed through design

Provide COMPREHENSIVE details in each category. Be as specific as possible - imagine you need to recreate this character exactly from text alone."""
                        
                        analysis_response = gemini_client.models.generate_content(
                            model=TEXT_MODEL,  # gemini-3-pro-preview
                            contents=[analysis_prompt, char_image]
                        )
                        
                        analysis_text = analysis_response.text if analysis_response.text else "Analysis failed"
                        
                        # 2단계: 부적 생성 프롬프트 작성
                        gemini_bujeok_prompt = locked_bujeok_prompt.format(
                            theme_name=selected_themes[1]['name'],
                            theme_keywords=selected_themes[1]['keywords']
                        )
                        
                        # 3단계: 완전한 text-to-image 프롬프트 생성 (캐릭터 재현 + 부적 변환)
                        from google.genai import types
                        full_prompt = f"""Create a vertical Korean fortune talisman (부적) artwork featuring this character.

CHARACTER DETAILS (You MUST recreate this character):
{analysis_text}

TALISMAN STYLE & THEME:
{gemini_bujeok_prompt}

COMPOSITION INSTRUCTIONS:
1. CENTER THE CHARACTER: Place the character in the center, recreating their appearance, pose, and clothing exactly as described.
2. TRANSFORM STYLE: Render the character with a 3D sculpted look, integrated into the talisman style.
3. TALISMAN ELEMENTS: Surround the character with traditional Korean talisman borders, red calligraphy-style symbols (abstract), and golden patterns.
4. THEME OBJECTS: Incorporate symbolic objects for {selected_themes[1]['name']} ({selected_themes[1]['keywords']}) around the character.
5. BACKGROUND: Aged yellow parchment texture with authentic Korean paper details.
6. ATMOSPHERE: Mystical, spiritual, dignified, and auspicious.

Negative Prompt: text, letters, watermarks, distorted face, bad anatomy, multiple characters, modern background."""
                        
                        # 4단계: Text-to-image 생성
                        response = gemini_client.models.generate_content(
                            model=IMAGE_MODEL,
                            contents=full_prompt,
                            config=types.GenerateContentConfig(
                                image_config=types.ImageConfig(
                                    aspect_ratio="9:16"
                                )
                            )
                        )
                        
                        gemini_img = None
                        if response and hasattr(response, 'candidates'):
                            for part in response.candidates[0].content.parts:
                                if hasattr(part, 'inline_data') and part.inline_data:
                                    img_bytes = part.inline_data.data
                                    gemini_img = Image.open(BytesIO(img_bytes)).convert("RGBA")
                                    break
                        
                        if gemini_img:
                            enhanced_results.append((
                                char_name,  # 캐릭터 이름
                                selected_themes[1]['name'],
                                "Gemini (캐릭터 부적)",
                                f"캐릭터 분석 기반 생성\n테마: {selected_themes[1]['name']} ({selected_themes[1]['keywords']})",
                                gemini_img
                            ))
                    except Exception as gemini_error:
                        print(f"Gemini 부적 생성 오류: {gemini_error}")
                
                if enhanced_results:
                    return {
                        "success": True, 
                        "results": enhanced_results, 
                        "valid_chars": selected_chars,
                        "char_count": len(valid_chars),
                        "error": None
                    }
                return {"success": False, "results": [], "valid_chars": [], "char_count": len(valid_chars), "error": "이미지 생성 실패"}
            return {"success": False, "results": [], "valid_chars": [], "char_count": len(valid_chars), "error": "캐릭터 이미지 또는 API 클라이언트 없음"}
        except Exception as e:
            import traceback
            return {"success": False, "results": [], "valid_chars": [], "char_count": 0, "error": f"{str(e)}\n{traceback.format_exc()}"}

    # 사주 이미지와 부적 이미지를 동시에 생성
    with st.spinner("🎨 사주 이미지와 부적 이미지를 동시에 생성 중... (병렬 처리)"):
        with ThreadPoolExecutor(max_workers=2) as executor:
            # 두 작업을 동시에 시작
            saju_future = executor.submit(generate_saju_image)
            bujeok_future = executor.submit(generate_bujeok_images_wrapper)
            
            # 결과 대기 (타임아웃 5분)
            try:
                saju_result = saju_future.result(timeout=300)
                if saju_result["success"]:
                    st.write("✅ 사주 이미지 생성 완료")
                    saju_img = saju_result["image"]
                else:
                    st.error(f"사주 이미지 생성 실패: {saju_result.get('error', '알 수 없는 오류')}")
                    saju_img = None
            except TimeoutError:
                st.error("사주 이미지 생성 타임아웃 (5분 초과)")
                saju_img = None
            except Exception as e:
                st.error(f"사주 이미지 처리 중 오류: {e}")
                saju_img = None
            
            try:
                bujeok_result = bujeok_future.result(timeout=300)
                if bujeok_result["success"]:
                    st.write(f"📂 발견된 캐릭터 이미지: {bujeok_result['char_count']}개")
                    st.write(f"✅ 부적 이미지 {len(bujeok_result['results'])}개 생성 완료")
                    bujeok_results_raw = bujeok_result["results"]
                    valid_chars = bujeok_result["valid_chars"]
                else:
                    st.warning(f"부적 이미지 생성 실패: {bujeok_result.get('error', '알 수 없는 오류')}")
                    bujeok_results_raw, valid_chars = [], []
            except TimeoutError:
                st.error("부적 이미지 생성 타임아웃 (5분 초과)")
                bujeok_results_raw, valid_chars = [], []
            except Exception as e:
                st.error(f"부적 이미지 처리 중 오류: {e}")
                bujeok_results_raw, valid_chars = [], []

    # 사주 이미지 처리
    if not saju_img:
        st.error("사주 이미지 생성에 실패했습니다.")
        st.stop()

    st.markdown("#### 🎨 생성된 사주 이미지")
    st.image(saju_img, caption="새해운세 이미지", use_container_width=True)

    # 이미지를 base64로 인코딩
    buffered = BytesIO()
    saju_img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    # 이미지 파일도 저장 (로컬 백업용)
    image_filename = f"saju_generated_{timestamp}.png"
    try:
        image_path = os.path.join(RESULT_DIR, image_filename)
        saju_img.save(image_path, format="PNG")
    except Exception as e:
        pass  # 파일 저장 실패는 무시

    # 부적 이미지 처리
    bujeok_results = []
    if bujeok_results_raw:
        st.markdown("#### 🧧 행운의 부적")
        
        # 2개의 부적 표시 (OpenAI, Gemini)
        cols = st.columns(2)
        for idx, (char_name, theme_name, model_name, prompt, img) in enumerate(bujeok_results_raw):
            if img:
                # base64로 인코딩
                bujeok_buffered = BytesIO()
                img.save(bujeok_buffered, format="PNG")
                img_b64 = base64.b64encode(bujeok_buffered.getvalue()).decode()
                bujeok_results.append((char_name, theme_name, model_name, img_b64))
                
                # 화면에 표시
                with cols[idx]:
                    st.markdown(f"**{theme_name} 부적**")
                    st.markdown(f"*{char_name} · {model_name}*")
                    st.image(img, use_container_width=True)
                    with st.expander("생성된 프롬프트"):
                        st.text(prompt if prompt else "프롬프트 생성 실패")
        
        if not bujeok_results:
            st.warning("부적 이미지 생성에 실패했습니다.")
    elif not valid_chars:
        st.info("img 폴더에 캐릭터 이미지(nana.png, Bbanya.png, Angmond.png)가 없습니다. 부적 생성을 건너뜁니다.")
    else:
        st.warning("부적 이미지 생성 중 오류가 발생했습니다.")

    # HTML 생성 - 섹션 키 매핑 (입력창 키 -> HTML 표시용 키)
    with st.spinner("📄 HTML 생성 중..."):
        # 디버깅: sections 딕셔너리의 모든 키 확인
        st.write("### 📋 입력된 sections 딕셔너리 키 확인")
        월별운_in_sections = [k for k in sections.keys() if '월별' in k]
        if 월별운_in_sections:
            st.write(f"✅ sections에 월별운 키 있음: {월별운_in_sections}")
            for k in 월별운_in_sections:
                st.write(f"  - 키: '{k}', 내용 길이: {len(sections[k])}자, 비어있음: {not sections[k].strip()}")
        else:
            st.warning(f"⚠️ sections에 월별운 키가 없습니다. 전체 키: {list(sections.keys())}")
        
        # 섹션 키를 HTML 생성 함수가 기대하는 형식으로 변환
        mapped_sections = {}
        for key, content in sections.items():
            # "(새해신수)", "(토정비결)" 등을 제거하여 간단한 키로 변환
            clean_key = key.replace("(새해신수)", "").replace("(토정비결)", "").replace(")", "")
            mapped_sections[clean_key] = content
        
        # 디버깅: 월별운 키와 내용 확인
        st.write("### 📋 변환된 mapped_sections 키 확인")
        월별운_keys = [k for k in mapped_sections.keys() if '월별' in k or '월별운' in k]
        if 월별운_keys:
            st.write(f"✅ 월별운 관련 키 발견: {월별운_keys}")
            for key in 월별운_keys:
                st.write(f"  - '{key}': {len(mapped_sections[key])}자, 키 표현: {repr(key)}")
        else:
            st.warning("⚠️ mapped_sections에 월별운 관련 키가 없습니다")
            st.write(f"사용 가능한 모든 키: {list(mapped_sections.keys())}")

        html_content = generate_html(
            user_name=user_name,
            gender=gender,
            solar_date=solar_date,
            lunar_date=lunar_date,
            birth_time=birth_time,
            sections=mapped_sections,
            image_base64=img_base64,
            chongun_summary=chongun_summary,
            bujeok_images=bujeok_results
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

    # 종료 시간 계산
    end_time = time.time()
    elapsed_time = end_time - start_time

    st.success(f"✅ HTML 생성 완료! (소요 시간: {elapsed_time:.1f}초)")

# 채팅방 요약 버튼 클릭 시
if generate_summary:
    # 시작 시간 기록
    summary_start_time = time.time()

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

                # 종료 시간 계산
                summary_end_time = time.time()
                summary_elapsed_time = summary_end_time - summary_start_time

                st.success(f"✅ 채팅방 요약 생성 완료! (소요 시간: {summary_elapsed_time:.1f}초)")
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
