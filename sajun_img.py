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
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

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
    "A prominent, large-scale close-up shot of a single instant Polaroid "
    "photograph held minimally by fingertips barely visible at the very "
    "bottom edge against a breathtaking seaside view. The Polaroid dominates "
    "the frame. The New Year's sun is just cresting the ocean horizon. The "
    "sky is bright and clear blues, casting brilliant morning light across "
    "the landscape. The photo displays a tiny, detailed diorama of the single "
    "person described in Scene Description, reimagined as a 3D chibi "
    "character. The classic white border of the Polaroid is completely blank, "
    "with no text or handwriting. Ethereal, clear morning glow illuminating "
    "the photo print, cinematic reflections on the glossy photo surface, cozy "
    "high-end aesthetic. Cinematic lighting, extremely shallow depth of field "
    "focusing sharply on the photo, ultra-polished photo paper texture, high "
    "detail, hopeful and whimsical New Year atmosphere. none text. Draw based "
    "on the following Scene Description, clearly specifying the gender."
)
DEFAULT_SUMMARY_INSTRUCTION = (
    "Read the provided Korean Saju text and create a vivid, single-scene "
    "description centered on the human figure that an image generation model "
    "can render as a beautiful painting.\n\n"
    "Your description MUST include the following:\n\n"
    "1. WHO (Core Subject): A specific human figure (gender must be clearly "
    "specified, depicted as a young adult in the prime of their life "
    "(approx. 20s) regardless of the age in the text, beautiful and elegant "
    "features, detailed attire, posture).\n\n"
    "2. WHAT (Core Action): A specific action or gesture the person is "
    "performing in that moment.\n\n"
    "3. WHERE (Background): A background that depicts the Saju's contents.\n\n"
    "The background must always be in Korea and include Korean cultural "
    "elements. (Women wear a skirt Hanbok, men wear pants Hanbok.)\n\n"
    "[MOST IMPORTANT INSTRUCTIONS]\n\n"
    "The absolute center of the description must always be the human figure.\n\n"
    "Irrespective of the age mentioned in the Saju text, the figure must "
    "strictly be described as young.\n\n"
    "Focus on positive, uplifting, and hopeful visual metaphors that inspire "
    "optimism and growth.\n\n"
    "Create the description without any sensitive content, such as pregnancy.\n\n"
    "Output the result in 1 English sentence."
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
- 1500자 내외로 요약
- 핵심 내용을 빠짐없이 전달하되 도사스러운 표현으로 재구성
- 맨 마지막에 더 자세히 보려면 신년운세 보기 버튼을 눌러보라고 안내해"""
DEFAULT_SCENE_SUMMARY_INSTRUCTION = """당신은 이미지 장면 설명과 운세 내용을 결합하여 한글로 간결하게 요약하는 전문가입니다.

요약 규칙:
- 정확히 5줄로 요약
- 장면의 시각적 요소와 운세의 핵심 메시지를 자연스럽게 융합
- 각 줄은 의미있는 핵심 포인트 하나씩
- 한글로 자연스럽게 표현
- 이모지 사용 금지
- 명확하고 구체적으로 도사말투로"""

# 부적 이미지 생성 프롬프트 (6개 테마별)
DEFAULT_BUJEOK_JEMUL = (
    "A traditional Korean yellow rectangular talisman with a red border on a red background. "
    "The bold red Korean text '영앤리치 인생한방' is at the top. "
    "Below it, the character from the reference image is wearing sunglasses and throwing money into the air "
    "with musical notes, money bags, and golden coins around them. "
    "The line art is thick, bold, and red in a woodblock print style."
)
DEFAULT_BUJEOK_YEONAE = (
    "A traditional Korean yellow rectangular talisman with a red border on a red background. "
    "The bold red Korean text '솔로탈출 인기폭발' is at the top. "
    "Below it, the character from the reference image is wearing sunglasses and making finger heart gestures "
    "surrounded by floating hearts, cupids, and roses. "
    "The line art is thick, bold, and red in a woodblock print style."
)
DEFAULT_BUJEOK_GUNGANG = (
    "A traditional Korean yellow rectangular talisman with a red border on a red background. "
    "The bold red Korean text '무병장수 천하무적' is at the top. "
    "Below it, the character from the reference image is wearing sunglasses and flexing their muscles "
    "showing strong energy, surrounded by energy shields and ginseng roots. "
    "The line art is thick, bold, and red in a woodblock print style."
)
DEFAULT_BUJEOK_JIKJANG = (
    "A traditional Korean yellow rectangular talisman with a red border on a red background. "
    "The bold red Korean text '초속승진 연봉떡상' is at the top. "
    "Below it, the character from the reference image is wearing sunglasses and sitting on a king's throne "
    "wearing a crown, surrounded by upward graph arrows and trophies. "
    "The line art is thick, bold, and red in a woodblock print style."
)
DEFAULT_BUJEOK_SOMANG = (
    "A traditional Korean yellow rectangular talisman with a red border on a red background. "
    "The bold red Korean text '소원성취 만사형통' is at the top. "
    "Below it, the character from the reference image is wearing sunglasses and holding a magical wishing lamp "
    "surrounded by sparkling stars and magic dust. "
    "The line art is thick, bold, and red in a woodblock print style."
)
DEFAULT_BUJEOK_ISA = (
    "A traditional Korean yellow rectangular talisman with a red border on a red background. "
    "The bold red Korean text '명당입성 대박기운' is at the top. "
    "Below it, the character from the reference image is wearing sunglasses and holding a golden key "
    "opening a new door, surrounded by swallows and lucky clouds. "
    "The line art is thick, bold, and red in a woodblock print style."
)

# ----------------------------
# 유틸
# ----------------------------
def get_gemini_client():
    if not GEMINI_API_KEY:
        return None
    try:
        # v1alpha API 버전 사용 (media_resolution 파라미터 지원)
        return genai.Client(
            api_key=GEMINI_API_KEY,
            http_options={'api_version': 'v1alpha'}
        )
    except Exception:
        return None

def get_openai_client():
    if not OPENAI_API_KEY or not OpenAI:
        return None
    try:
        # httpx 클라이언트로 프록시 환경 변수 자동 적용 및 타임아웃 설정
        try:
            import httpx
            # trust_env=True로 HTTP_PROXY, HTTPS_PROXY 환경 변수 사용
            # 타임아웃: 연결 60초, 읽기 150초 (이미지 생성 평균 30초~2분)
            http_client = httpx.Client(
                trust_env=True,
                timeout=httpx.Timeout(connect=60.0, read=150.0, write=60.0, pool=60.0)
            )
            client = OpenAI(api_key=OPENAI_API_KEY, http_client=http_client)
            return client
        except ImportError:
            # httpx가 없으면 기본 클라이언트 사용 (타임아웃 150초)
            client = OpenAI(api_key=OPENAI_API_KEY, timeout=150.0)
            return client
    except Exception as e:
        st.warning(f"OpenAI 클라이언트 초기화 실패: {e}")
        return None


def summarize_to_three_lines(
    source_text: str,
    openai_client: Optional[OpenAI] = None,
) -> str:
    """
    텍스트를 5줄로 요약
    """
    system_instruction = """당신은 사주 내용을 간결하게 요약하는 전문가입니다.

요약 규칙:
- 정확히 5줄로 요약
- 각 줄은 핵심 포인트 하나씩
- 간결하고 명확하게
- 이모지 사용 금지"""

    user_msg = f"""다음 총운 내용을 정확히 5줄로 요약해주세요:

{source_text}

[요구사항]
- 5줄로 요약
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

def summarize_scene_to_korean_three_lines(
    scene_text: str,
    chongun_text: str = "",
    openai_client: Optional[OpenAI] = None,
    system_instruction: str = DEFAULT_SCENE_SUMMARY_INSTRUCTION,
) -> str:
    """
    영문 장면 요약과 총운 내용을 함께 활용하여 한글로 요약 (줄 수는 system_instruction에서 동적으로 결정)
    """
    # system_instruction에서 줄 수 추출 (예: "정확히 5줄로 요약" -> 5)
    import re
    line_match = re.search(r'정확히 (\d+)줄로 요약', system_instruction)
    line_count = int(line_match.group(1)) if line_match else 5  # 기본값 5줄

    if chongun_text:
        user_msg = f"""다음 이미지 장면 설명과 총운 내용을 함께 고려하여 한글로 정확히 {line_count}줄로 요약해주세요:

[이미지 장면 설명]
{scene_text}

[총운 내용]
{chongun_text}

[요구사항]
- 한글로 {line_count}줄 요약
- 각 줄은 한 문장
- 장면의 시각적 요소와 운세의 핵심을 자연스럽게 결합
- 독자가 이미지와 운세의 연결고리를 이해할 수 있도록"""
    else:
        user_msg = f"""다음 이미지 장면 설명을 한글로 정확히 {line_count}줄로 요약해주세요:

{scene_text}

[요구사항]
- 한글로 {line_count}줄 요약
- 각 줄은 한 문장
- 시각적 핵심 요소만 전달
- 자연스러운 한국어 표현"""

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
                    img_bytes = requests.get(img_data.url, timeout=120).content

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
            # Gemini 3 권장사항: temperature=1.0 유지
            final_prompt = f"Create a picture of: {prompt} (Aspect Ratio: 9:16)"
            
            response = gemini_client.models.generate_content(
                model=IMAGE_MODEL,
                contents=final_prompt,
                config=types.GenerateContentConfig(
                    temperature=1.0
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
    """프롬프트로 단일 부적 이미지를 생성하는 함수"""
    import sys
    print(f"[부적생성] 이미지 파일 열기 시작: {image_path}", file=sys.stderr)
    
    # images.edit 사용하여 캐릭터 보존하면서 스타일 변경
    with open(image_path, "rb") as img_file:
        print(f"[부적생성] OpenAI API 호출 시작 (images.edit)", file=sys.stderr)
        response = openai_client.images.edit(
            model="gpt-image-1",
            image=img_file,
            prompt=prompt,
            n=1,
            size="1024x1536"
        )
        print(f"[부적생성] OpenAI API 응답 받음", file=sys.stderr)
    
    if response.data:
        img_data = response.data[0]
        print(f"[부적생성] 이미지 데이터 추출 중", file=sys.stderr)
        if getattr(img_data, "url", None):
            print(f"[부적생성] URL에서 이미지 다운로드 중", file=sys.stderr)
            image_bytes = requests.get(img_data.url, timeout=60).content
        else:
            print(f"[부적생성] base64 디코딩 중", file=sys.stderr)
            image_bytes = base64.b64decode(img_data.b64_json)
        
        print(f"[부적생성] PIL 이미지 변환 중", file=sys.stderr)
        img = Image.open(BytesIO(image_bytes)).convert("RGBA")
        print(f"[부적생성] 부적 이미지 생성 완료!", file=sys.stderr)
        return img
    
    print(f"[부적생성] 응답 데이터가 없음", file=sys.stderr)
    return None

def generate_bujeok_images(base_prompt: str, char_images: list, openai_client: OpenAI):
    """
    여러 캐릭터 이미지로 부적 이미지들을 순차적으로 생성
    char_images: [(name, path), ...] 형식의 리스트
    반환: [(name, prompt, image), ...] 형식의 리스트
    """
    import sys
    results = []
    
    print(f"[부적생성] 총 {len(char_images)}개 캐릭터 부적 생성 시작", file=sys.stderr)
    
    # 순차적으로 이미지 생성 (Streamlit 안정성 확보)
    for idx, (char_name, img_path) in enumerate(char_images, 1):
        try:
            print(f"[부적생성] {idx}/{len(char_images)}: {char_name} 부적 생성 시작", file=sys.stderr)
            img = generate_bujeok_image_single(base_prompt, img_path, openai_client)
            results.append((char_name, base_prompt, img))
            print(f"[부적생성] {idx}/{len(char_images)}: {char_name} 부적 생성 완료", file=sys.stderr)
        except Exception as exc:
            # 에러 발생 시에도 계속 진행
            import traceback
            print(f"⚠️ {char_name} 부적 생성 실패: {exc}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            results.append((char_name, base_prompt, None))
    
    print(f"[부적생성] 전체 부적 생성 완료: {len(results)}개", file=sys.stderr)
    return results

def generate_html(user_name: str, gender: str, solar_date: str, lunar_date: str,
                  birth_time: str, sections: dict, image_base64: str,
                  chongun_summary: str = "", bujeok_images: list = None,
                  timing_info: dict = None) -> str:
    """
    19개 섹션 내용을 받아서 HTML을 생성
    image_base64: base64로 인코딩된 이미지 데이터
    chongun_summary: 장면 요약 + 총운 내용 한글 3줄 정리
    bujeok_images: 부적 이미지 리스트 [(char_name, theme_name, model_name, base64), ...]
    """
    # 디버깅: HTML 생성 함수에 전달된 sections 확인 (주석 처리 - 필요시 활성화)
    # import sys
    # print(f"[HTML DEBUG] generate_html 함수 시작", file=sys.stderr)
    # print(f"[HTML DEBUG] sections 키 목록: {list(sections.keys())}", file=sys.stderr)
    
    if bujeok_images is None:
        bujeok_images = []
    if timing_info is None:
        timing_info = {}
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
            # 장면 요약 + 총운 3줄을 이미지 위에 표시
            if chongun_summary:
                html += f"""
                <!-- 핵심 장면 + 총운 3줄 요약 -->
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
                    <div class="flex flex-col items-center justify-center mt-8 gap-8">
"""
        for char_name, theme_name, model_name, img_base64 in bujeok_images:
            html += f"""
                        <div class="flex flex-col items-center max-w-md w-full">
                            <h3 class="text-xl font-semibold text-gray-800 mb-2">{theme_name} 부적</h3>
                            <p class="text-sm text-gray-600 mb-4">{char_name} · {model_name}</p>
                            <img src="data:image/png;base64,{img_base64}" alt="{theme_name} 부적" class="rounded-lg shadow-xl w-full h-auto">
                        </div>
"""
        html += """                    </div>
                </div>
            </section>
"""

    html += """        </div>
    </main>

    <!-- 단계별 소요시간 정보 -->
    <div class="mt-12 p-6 bg-gray-50 rounded-lg border border-gray-200">
        <h3 class="text-lg font-semibold text-gray-800 mb-4 flex items-center">
            <span class="mr-2">⏱️</span>
            생성 단계별 소요시간
        </h3>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
"""

    # 단계별 시간 정보를 HTML에 추가
    timing_items = [
        ("텍스트 분석 및 섹션 매핑", timing_info.get("text_analysis", 0)),
        ("총운 5줄 요약 생성", timing_info.get("chongun_summary", 0)),
        ("장면 5줄 요약 생성", timing_info.get("scene_summary", 0)),
        ("사주 이미지 생성", timing_info.get("saju_image", 0)),
        ("부적 이미지 생성", timing_info.get("bujeok_image", 0)),
        ("HTML 생성", timing_info.get("html_generation", 0)),
    ]

    total_time = sum(time for _, time in timing_items)

    for step_name, step_time in timing_items:
        if step_time > 0:
            percentage = (step_time / total_time * 100) if total_time > 0 else 0
            html += f"""            <div class="flex justify-between items-center p-3 bg-white rounded border">
                <span class="text-sm font-medium text-gray-700">{step_name}</span>
                <div class="flex items-center space-x-2">
                    <div class="w-16 bg-gray-200 rounded-full h-2">
                        <div class="bg-blue-500 h-2 rounded-full" style="width: {percentage:.1f}%"></div>
                    </div>
                    <span class="text-sm text-gray-600 min-w-[60px]">{step_time:.1f}초</span>
                </div>
            </div>
"""

    html += f"""            <div class="col-span-full mt-4 pt-4 border-t border-gray-300">
                <div class="flex justify-between items-center p-3 bg-blue-50 rounded border border-blue-200">
                    <span class="text-sm font-semibold text-blue-800">전체 소요시간</span>
                    <span class="text-sm font-semibold text-blue-800">{total_time:.1f}초</span>
                </div>
            </div>
        </div>
        <div class="mt-4 text-xs text-gray-500 text-center">
            생성 시각: {timing_info.get("generated_at", "알 수 없음")}
        </div>
    </div>
"""

    # JavaScript 부분을 별도로 추가 (f-string 문제 회피)
    html += """    <script>
        /* 앵커 링크 클릭 시 스크롤만 처리 (페이지 리로드 방지) */
        document.addEventListener('DOMContentLoaded', function() {
            /* 모든 앵커 링크에 이벤트 리스너 추가 */
            document.querySelectorAll('a[href^="#"]').forEach(anchor => {
                anchor.addEventListener('click', function(e) {
                    e.preventDefault(); /* 기본 동작 방지 */

                    const targetId = this.getAttribute('href').substring(1);
                    const targetElement = document.getElementById(targetId);

                    if (targetElement) {
                        /* 부드러운 스크롤 */
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

# 세션 상태 초기화 (안전하게 처리)
try:
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
except Exception as e:
    st.error(f"세션 상태 초기화 중 오류가 발생했습니다: {e}")
    st.stop()

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

st.markdown("---")
st.markdown("### 🧧 부적 이미지 생성 프롬프트 (테마별)")

bujeok_jemul_input = st.text_area(
    "재물운 부적 프롬프트",
    value=DEFAULT_BUJEOK_JEMUL,
    height=100,
    help="재물운 부적 이미지 생성 프롬프트입니다.",
)
bujeok_jemul = bujeok_jemul_input if bujeok_jemul_input.strip() else DEFAULT_BUJEOK_JEMUL

bujeok_yeonae_input = st.text_area(
    "연애운 부적 프롬프트",
    value=DEFAULT_BUJEOK_YEONAE,
    height=100,
    help="연애운 부적 이미지 생성 프롬프트입니다.",
)
bujeok_yeonae = bujeok_yeonae_input if bujeok_yeonae_input.strip() else DEFAULT_BUJEOK_YEONAE

bujeok_gungang_input = st.text_area(
    "건강운 부적 프롬프트",
    value=DEFAULT_BUJEOK_GUNGANG,
    height=100,
    help="건강운 부적 이미지 생성 프롬프트입니다.",
)
bujeok_gungang = bujeok_gungang_input if bujeok_gungang_input.strip() else DEFAULT_BUJEOK_GUNGANG

bujeok_jikjang_input = st.text_area(
    "직장운 부적 프롬프트",
    value=DEFAULT_BUJEOK_JIKJANG,
    height=100,
    help="직장운 부적 이미지 생성 프롬프트입니다.",
)
bujeok_jikjang = bujeok_jikjang_input if bujeok_jikjang_input.strip() else DEFAULT_BUJEOK_JIKJANG

bujeok_somang_input = st.text_area(
    "소망운 부적 프롬프트",
    value=DEFAULT_BUJEOK_SOMANG,
    height=100,
    help="소망운 부적 이미지 생성 프롬프트입니다.",
)
bujeok_somang = bujeok_somang_input if bujeok_somang_input.strip() else DEFAULT_BUJEOK_SOMANG

bujeok_isa_input = st.text_area(
    "이사운 부적 프롬프트",
    value=DEFAULT_BUJEOK_ISA,
    height=100,
    help="이사운 부적 이미지 생성 프롬프트입니다.",
)
bujeok_isa = bujeok_isa_input if bujeok_isa_input.strip() else DEFAULT_BUJEOK_ISA

st.markdown("---")

chat_summary_prompt_input = st.text_area(
    "채팅방 요약 시스템 프롬프트",
    value=DEFAULT_CHAT_SUMMARY_INSTRUCTION,
    height=150,
    help="채팅방 요약 생성 모델에 전달할 시스템 메시지입니다. {user_name}은 자동으로 치환됩니다.",
)
chat_summary_prompt = chat_summary_prompt_input if chat_summary_prompt_input.strip() else DEFAULT_CHAT_SUMMARY_INSTRUCTION

scene_summary_prompt_input = st.text_area(
    "사주 이미지 설명 프롬프트",
    value=DEFAULT_SCENE_SUMMARY_INSTRUCTION,
    height=150,
    help="이미지 장면 설명과 총운 내용을 결합하여 한글 설명을 생성할 때 사용하는 시스템 프롬프트입니다.",
)
scene_summary_prompt = scene_summary_prompt_input if scene_summary_prompt_input.strip() else DEFAULT_SCENE_SUMMARY_INSTRUCTION


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

    # 단계별 소요시간 기록용 딕셔너리
    timing_info = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    }

    # "올해의총운" 텍스트로 이미지 생성
    base_text = sections.get("올해의총운(새해신수)", "").strip()
    if not base_text:
        st.error("'올해의총운'을 입력해주세요. 이 내용으로 이미지를 생성합니다.")
        st.stop()

    # 이미지 생성 시작 시점의 설정을 고정
    locked_system_prompt = system_prompt
    locked_summary_prompt = summary_prompt
    locked_bujeok_prompts = {
        "재물운": bujeok_jemul,
        "연애운": bujeok_yeonae,
        "건강운": bujeok_gungang,
        "직장운": bujeok_jikjang,
        "소망운": bujeok_somang,
        "이사운": bujeok_isa,
    }
    locked_chat_summary_prompt = chat_summary_prompt
    locked_scene_summary_prompt = scene_summary_prompt
    locked_openai_client = openai_client

    # 진행 상황 로그 컨테이너
    progress_log = st.empty()
    
    progress_log.info("🔄 1/6 단계: 핵심 장면 추출 중...")
    text_analysis_start = time.time()
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

    text_analysis_end = time.time()
    timing_info["text_analysis"] = text_analysis_end - text_analysis_start

    core_scene = (core_scene or "").strip()
    st.session_state["core_scene_summary"] = core_scene
    if core_scene:
        st.markdown("#### ✨ 핵심 장면 요약")
        st.write(core_scene)

    progress_log.success("✅ 1/6 단계 완료: 핵심 장면 추출")

    # 장면 요약과 총운 내용을 함께 한글 3줄로 정리
    progress_log.info("🔄 2/6 단계: 장면 요약 + 총운 한글 3줄 정리 중...")
    scene_summary_start = time.time()
    with st.spinner("📋 장면 요약 정리 중 (gpt-4.1-mini 사용)..."):
        try:
            chongun_text = sections.get("핵심포인트(새해신수)", "").strip() + "\n\n" + sections.get("올해의총운(새해신수)", "").strip()
            scene_summary_korean = summarize_scene_to_korean_three_lines(
                scene_text=core_scene,
                chongun_text=chongun_text,
                openai_client=locked_openai_client,
                system_instruction=locked_scene_summary_prompt
            )
        except Exception as exc:
            st.warning(f"장면 요약 정리 중 오류: {exc}")
            scene_summary_korean = ""

    scene_summary_end = time.time()
    timing_info["scene_summary"] = scene_summary_end - scene_summary_start

    progress_log.success("✅ 2/6 단계 완료: 장면 요약 + 총운 한글 3줄 정리")

    progress_log.info("🔄 3/6 단계: 이미지 프롬프트 작성 중...")
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
    
    progress_log.success("✅ 3/6 단계 완료: 이미지 프롬프트 작성")

    final_prompt = prompt
    timestamp = int(time.time())

    # 사주 이미지 생성 함수
    def generate_saju_image():
        saju_start_time = time.time()
        try:
            imgs = generate_images(
                final_prompt,
                num_images=1,
                provider="openai",
                gemini_client=None,
                openai_client=locked_openai_client,
            )
            valid = [i for i in imgs if i is not None]
            saju_end_time = time.time()
            saju_elapsed = saju_end_time - saju_start_time
            return {"success": True, "image": valid[0] if valid else None, "error": None, "elapsed_time": saju_elapsed}
        except Exception as e:
            saju_end_time = time.time()
            saju_elapsed = saju_end_time - saju_start_time
            return {"success": False, "image": None, "error": str(e), "elapsed_time": saju_elapsed}

    # 부적 이미지 생성 함수 (OpenAI 단독 생성)
    def generate_bujeok_images_wrapper():
        import sys
        bujeok_start_time = time.time()
        try:
            print("[부적Wrapper] 부적 생성 시작", file=sys.stderr)
            import random
            img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img")
            char_images = [
                ("나나", os.path.join(img_dir, "nana.png")),
                ("뱐냐", os.path.join(img_dir, "Bbanya.png")),
                ("앙몬드", os.path.join(img_dir, "Angmond.png"))
            ]
            
            print(f"[부적Wrapper] 캐릭터 이미지 경로 확인 중...", file=sys.stderr)
            valid_chars = [(name, path) for name, path in char_images if os.path.exists(path)]
            print(f"[부적Wrapper] 발견된 캐릭터: {len(valid_chars)}개 - {[name for name, _ in valid_chars]}", file=sys.stderr)
            
            if valid_chars and locked_openai_client:
                # 랜덤으로 캐릭터 1개 선택
                selected_chars = random.sample(valid_chars, 1)
                print(f"[부적Wrapper] 선택된 캐릭터: {selected_chars[0][0]}", file=sys.stderr)
                
                # UI에서 설정한 프롬프트로 themes 배열 구성
                themes = [
                    {"name": theme_name, "prompt": prompt}
                    for theme_name, prompt in locked_bujeok_prompts.items()
                ]
                selected_themes = random.sample(themes, 1)
                print(f"[부적Wrapper] 선택된 테마: {selected_themes[0]['name']}", file=sys.stderr)
                
                enhanced_results = []
                
                # OpenAI로 부적 생성 (캐릭터 부적 - 이미지 편집)
                print(f"[부적Wrapper] 부적 프롬프트 생성 중...", file=sys.stderr)
                openai_prompt = selected_themes[0]['prompt']
                print(f"[부적Wrapper] generate_bujeok_images() 호출", file=sys.stderr)
                openai_results = generate_bujeok_images(openai_prompt, [selected_chars[0]], locked_openai_client)
                print(f"[부적Wrapper] generate_bujeok_images() 완료, 결과 개수: {len(openai_results)}", file=sys.stderr)
                
                if openai_results and openai_results[0][2] is not None:
                    print(f"[부적Wrapper] 부적 이미지 생성 성공!", file=sys.stderr)
                    enhanced_results.append((
                        openai_results[0][0],  # 캐릭터 이름
                        selected_themes[0]['name'], 
                        "OpenAI (캐릭터 부적)",
                        openai_results[0][1], 
                        openai_results[0][2]
                    ))
                else:
                    print(f"[부적Wrapper] 부적 이미지가 None입니다", file=sys.stderr)
                
                bujeok_end_time = time.time()
                bujeok_elapsed = bujeok_end_time - bujeok_start_time
                
                if enhanced_results:
                    print(f"[부적Wrapper] 최종 결과: 성공 ({len(enhanced_results)}개)", file=sys.stderr)
                    return {
                        "success": True, 
                        "results": enhanced_results, 
                        "valid_chars": selected_chars,
                        "char_count": len(valid_chars),
                        "error": None,
                        "logs": [],
                        "elapsed_time": bujeok_elapsed
                    }
                
                print(f"[부적Wrapper] 최종 결과: 실패 (enhanced_results가 비어있음)", file=sys.stderr)
                return {"success": False, "results": [], "valid_chars": [], "char_count": len(valid_chars), "error": "OpenAI 이미지 생성 실패", "logs": [], "elapsed_time": bujeok_elapsed}
            
            bujeok_end_time = time.time()
            bujeok_elapsed = bujeok_end_time - bujeok_start_time
            print(f"[부적Wrapper] valid_chars 또는 openai_client가 없음", file=sys.stderr)
            return {"success": False, "results": [], "valid_chars": [], "char_count": 0, "error": "캐릭터 이미지 또는 OpenAI 클라이언트 없음", "logs": [], "elapsed_time": bujeok_elapsed}
        except Exception as e:
            import traceback
            bujeok_end_time = time.time()
            bujeok_elapsed = bujeok_end_time - bujeok_start_time
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            print(f"[부적Wrapper] 예외 발생: {error_msg}", file=sys.stderr)
            return {"success": False, "results": [], "valid_chars": [], "char_count": 0, "error": error_msg, "logs": [], "elapsed_time": bujeok_elapsed}

    # 사주 이미지와 부적 이미지를 순차적으로 생성 (안정성 확보 및 디버깅 용이)
    # 병렬 처리 시 원인 불명의 중단 현상이 발생하여 순차 처리로 변경함
    
    # 4-5. 사주 이미지와 부적 이미지 동시 생성 (병렬 처리)
    import sys
    progress_log.info("🔄 4-5/6 단계: 사주 이미지와 부적 이미지를 동시에 생성 중...")
    print("[병렬생성] 사주 + 부적 이미지 동시 생성 시작", file=sys.stderr)

    saju_img = None
    saju_error = None
    bujeok_results_raw = []
    valid_chars = []
    bujeok_status = None
    bujeok_error = None

    # 이미지 생성 시작 시간 기록
    image_generation_start = time.time()

    with st.spinner("🎨 사주 이미지와 부적 이미지를 동시에 생성 중입니다..."):
        with ThreadPoolExecutor(max_workers=2) as executor:
            print("[병렬생성] ThreadPoolExecutor 시작 (워커 2개)", file=sys.stderr)
            
            # 두 작업을 동시에 제출
            print("[병렬생성] 사주 이미지 생성 작업 제출", file=sys.stderr)
            future_saju = executor.submit(generate_saju_image)
            
            print("[병렬생성] 부적 이미지 생성 작업 제출", file=sys.stderr)
            future_bujeok = executor.submit(generate_bujeok_images_wrapper)
            
            # 작업 완료 대기 및 결과 수집
            print("[병렬생성] 작업 완료 대기 중...", file=sys.stderr)
            futures = {
                future_saju: "사주",
                future_bujeok: "부적"
            }
            
            try:
                for future in as_completed(futures, timeout=360):  # 전체 6분 타임아웃
                    task_name = futures[future]
                    try:
                        print(f"[병렬생성] {task_name} 작업 완료됨", file=sys.stderr)
                        
                        if task_name == "사주":
                            saju_result = future.result(timeout=180)  # 사주 이미지 최대 3분
                            print(f"[병렬생성] 사주 결과 획득: success={saju_result.get('success')}", file=sys.stderr)
                            if saju_result["success"]:
                                saju_img = saju_result["image"]
                                saju_elapsed = saju_result.get("elapsed_time", 0)
                                timing_info["saju_image"] = saju_elapsed
                                print(f"[병렬생성] 사주 이미지 저장 완료 (소요시간: {saju_elapsed:.1f}초)", file=sys.stderr)
                            else:
                                saju_error = f"사주 이미지 생성 실패: {saju_result.get('error', '알 수 없는 오류')}"
                                saju_elapsed = saju_result.get("elapsed_time", 0)
                                timing_info["saju_image"] = saju_elapsed
                                print(f"[병렬생성] {saju_error} (소요시간: {saju_elapsed:.1f}초)", file=sys.stderr)
                        
                        elif task_name == "부적":
                            bujeok_result = future.result(timeout=180)  # 부적 이미지 최대 3분
                            print(f"[병렬생성] 부적 결과 획득: success={bujeok_result.get('success')}", file=sys.stderr)
                            if bujeok_result["success"]:
                                bujeok_results_raw = bujeok_result["results"]
                                valid_chars = bujeok_result["valid_chars"]
                                bujeok_status = f"✅ 부적 이미지 {len(bujeok_result['results'])}개 생성 완료"
                                bujeok_elapsed = bujeok_result.get("elapsed_time", 0)
                                timing_info["bujeok_image"] = bujeok_elapsed
                                print(f"[병렬생성] 부적 결과 저장 완료: {len(bujeok_results_raw)}개 (소요시간: {bujeok_elapsed:.1f}초)", file=sys.stderr)
                            else:
                                bujeok_error = f"부적 이미지 생성 실패: {bujeok_result.get('error', '알 수 없는 오류')}"
                                bujeok_elapsed = bujeok_result.get("elapsed_time", 0)
                                timing_info["bujeok_image"] = bujeok_elapsed
                                print(f"[병렬생성] {bujeok_error} (소요시간: {bujeok_elapsed:.1f}초)", file=sys.stderr)
                    
                    except TimeoutError as e:
                        timeout_msg = f"{task_name} 작업 타임아웃 (3분 초과): {e}"
                        print(f"[병렬생성] ⏱️ {timeout_msg}", file=sys.stderr)
                        
                        if task_name == "사주":
                            saju_error = timeout_msg
                        elif task_name == "부적":
                            bujeok_error = timeout_msg
                    
                    except Exception as e:
                        import traceback
                        error_msg = f"{task_name} 작업 중 예외: {e}\n{traceback.format_exc()}"
                        print(f"[병렬생성] {error_msg}", file=sys.stderr)
                        
                        if task_name == "사주":
                            saju_error = error_msg
                        elif task_name == "부적":
                            bujeok_error = error_msg
            
            except TimeoutError:
                # as_completed 전체 타임아웃
                overall_timeout_msg = "⏱️ 전체 작업 타임아웃 (6분 초과) - 일부 작업이 완료되지 못했습니다"
                print(f"[병렬생성] {overall_timeout_msg}", file=sys.stderr)
                if not saju_img:
                    saju_error = "사주 이미지 생성 타임아웃"
                if not bujeok_status:
                    bujeok_error = "부적 이미지 생성 타임아웃"
            
            print("[병렬생성] 모든 작업 완료, ThreadPoolExecutor 종료", file=sys.stderr)
    
    print("[병렬생성] 스피너 종료, 결과 확인", file=sys.stderr)
    
    # 결과 표시
    if saju_img and (bujeok_status or bujeok_error):
        progress_log.success("✅ 4-5/6 단계 완료: 사주 이미지 및 부적 이미지 생성 완료")
        print("[병렬생성] 양쪽 작업 모두 완료", file=sys.stderr)
    elif saju_img:
        progress_log.success("✅ 4-5/6 단계 완료: 사주 이미지 생성 완료 (부적은 실패)")
        if bujeok_error:
            st.warning(bujeok_error)
        print("[병렬생성] 사주만 성공, 부적 실패", file=sys.stderr)
    elif saju_error:
        st.error(saju_error)
        print("[병렬생성] 사주 생성 실패", file=sys.stderr)
        st.stop()
    
    # 이미지 생성 완료 시간 기록 (각 작업의 시간은 이미 timing_info에 저장됨)
    image_generation_end = time.time()
    total_image_time = image_generation_end - image_generation_start
    
    # 각 작업의 시간이 측정되지 않은 경우에만 대략적으로 분배
    if "saju_image" not in timing_info or timing_info["saju_image"] == 0:
        timing_info["saju_image"] = total_image_time * 0.6  # 사주 이미지 시간
    if "bujeok_image" not in timing_info or timing_info["bujeok_image"] == 0:
        timing_info["bujeok_image"] = total_image_time * 0.4  # 부적 이미지 시간

    print(f"[병렬생성] 병렬 생성 단계 완전 종료 (전체: {total_image_time:.1f}초, 사주: {timing_info.get('saju_image', 0):.1f}초, 부적: {timing_info.get('bujeok_image', 0):.1f}초)", file=sys.stderr)

    # 사주 이미지 처리
    import sys
    print("[UI] 사주 이미지 처리 시작", file=sys.stderr)
    
    if not saju_img:
        st.error("사주 이미지 생성에 실패했습니다.")
        st.stop()

    # 부적 이미지 처리 (먼저 데이터 준비)
    print("[UI] 부적 이미지 처리 시작", file=sys.stderr)
    bujeok_results = []
    bujeok_img_to_display = None
    bujeok_theme_name = None
    bujeok_char_name = None
    bujeok_model_name = None
    bujeok_prompt = None
    
    if bujeok_results_raw:
        try:
            print(f"[UI] 부적 {len(bujeok_results_raw)}개 처리 시작", file=sys.stderr)
            
            # 1개의 부적 표시 (OpenAI)
            for idx, (char_name, theme_name, model_name, prompt, img) in enumerate(bujeok_results_raw, 1):
                print(f"[UI] 부적 {idx} 처리: {char_name} - {theme_name}", file=sys.stderr)
                if img:
                    # base64로 인코딩
                    print(f"[UI] 부적 {idx} base64 인코딩", file=sys.stderr)
                    bujeok_buffered = BytesIO()
                    img.save(bujeok_buffered, format="PNG")
                    img_b64 = base64.b64encode(bujeok_buffered.getvalue()).decode()
                    bujeok_results.append((char_name, theme_name, model_name, img_b64))
                    print(f"[UI] 부적 {idx} 인코딩 완료: {len(img_b64)} 문자", file=sys.stderr)
                    
                    # 첫 번째 부적만 화면에 표시할 준비
                    if bujeok_img_to_display is None:
                        bujeok_img_to_display = img
                        bujeok_theme_name = theme_name
                        bujeok_char_name = char_name
                        bujeok_model_name = model_name
                        bujeok_prompt = prompt
                        print(f"[UI] 부적 {idx} 표시 준비 완료", file=sys.stderr)
        except Exception as e:
            import traceback
            error_msg = f"부적 이미지 처리 중 오류: {e}\n{traceback.format_exc()}"
            print(f"[UI] 부적 처리 예외: {error_msg}", file=sys.stderr)
            st.error(error_msg)

    # 사주 이미지와 부적을 한 행에 반반씩 표시
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            print("[UI] 사주 이미지 화면 표시", file=sys.stderr)
            st.markdown("#### 🎨 생성된 사주 이미지")
            st.image(saju_img, caption="새해운세 이미지", use_container_width=True)
            print("[UI] 사주 이미지 표시 완료", file=sys.stderr)
        except Exception as e:
            print(f"[UI] 사주 이미지 표시 실패: {e}", file=sys.stderr)
            st.error(f"이미지 표시 중 오류: {e}")
    
    with col2:
        if bujeok_img_to_display:
            print("[UI] 부적 이미지 화면 표시 시작", file=sys.stderr)
            st.markdown("#### 🧧 행운의 부적")
            st.markdown(f"**{bujeok_theme_name} 부적**")
            st.markdown(f"*{bujeok_char_name} · {bujeok_model_name}*")
            st.image(bujeok_img_to_display, use_container_width=True)
            with st.expander("생성된 프롬프트"):
                st.text(bujeok_prompt if bujeok_prompt else "프롬프트 생성 실패")
            print("[UI] 부적 이미지 화면 표시 완료", file=sys.stderr)
        elif bujeok_results_raw and not bujeok_results:
            st.warning("부적 이미지 생성에 실패했습니다.")
            print("[UI] 부적 결과가 비어있음", file=sys.stderr)
        elif not valid_chars:
            st.info("img 폴더에 캐릭터 이미지(nana.png, Bbanya.png, Angmond.png)가 없습니다. 부적 생성을 건너뜁니다.")
            print("[UI] 캐릭터 이미지 없음", file=sys.stderr)
        else:
            st.warning("부적 이미지 생성 중 오류가 발생했습니다.")
            print("[UI] 부적 생성 오류", file=sys.stderr)
    
    # 이미지를 base64로 인코딩 (HTML 생성용)
    print("[UI] 사주 이미지 base64 인코딩 시작", file=sys.stderr)
    buffered = BytesIO()
    saju_img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    print(f"[UI] base64 인코딩 완료: {len(img_base64)} 문자", file=sys.stderr)

    # 이미지 파일도 저장 (로컬 백업용)
    image_filename = f"saju_generated_{timestamp}.png"
    try:
        image_path = os.path.join(RESULT_DIR, image_filename)
        saju_img.save(image_path, format="PNG")
        print("[UI] 사주 이미지 파일 저장 완료", file=sys.stderr)
    except Exception as e:
        print(f"[UI] 사주 이미지 파일 저장 실패 (무시): {e}", file=sys.stderr)
    
    print("[UI] 부적 이미지 처리 완료, 6단계로 진행", file=sys.stderr)

    # 6. HTML 생성
    import sys
    progress_log.info("🔄 6/6 단계: HTML 생성 중...")
    print("[UI] 6단계 시작: HTML 생성", file=sys.stderr)

    html_generation_start = time.time()

    html_content = None
    html_filename = None

    with st.spinner("📄 HTML 생성 중..."):
        try:
            print("[UI] 섹션 키 매핑 시작", file=sys.stderr)
            # 섹션 키를 HTML 생성 함수가 기대하는 형식으로 변환
            mapped_sections = {}
            for key, content in sections.items():
                # "(새해신수)", "(토정비결)" 등을 제거하여 간단한 키로 변환
                clean_key = key.replace("(새해신수)", "").replace("(토정비결)", "").replace(")", "")
                mapped_sections[clean_key] = content
            
            print(f"[UI] 섹션 매핑 완료: {len(mapped_sections)}개", file=sys.stderr)
            print(f"[UI] 부적 이미지 개수: {len(bujeok_results)}", file=sys.stderr)
            print("[UI] generate_html() 호출", file=sys.stderr)
            
            html_content = generate_html(
                user_name=user_name,
                gender=gender,
                solar_date=solar_date,
                lunar_date=lunar_date,
                birth_time=birth_time,
                sections=mapped_sections,
                image_base64=img_base64,
                chongun_summary=scene_summary_korean,
                bujeok_images=bujeok_results,
                timing_info=timing_info
            )
            
            print(f"[UI] HTML 생성 완료: {len(html_content)} 문자", file=sys.stderr)
            
            html_filename = f"{user_name}_tojeung_{timestamp}.html"

            # 파일 저장 시도 (실패해도 계속 진행)
            try:
                print("[UI] HTML 파일 저장 시도", file=sys.stderr)
                html_path = os.path.join(RESULT_DIR, html_filename)
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(html_content)
                print("[UI] HTML 파일 저장 완료", file=sys.stderr)
            except Exception as e:
                print(f"[UI] HTML 파일 저장 실패 (무시): {e}", file=sys.stderr)
                
        except Exception as e:
            import traceback
            error_msg = f"HTML 생성 중 오류: {e}\n{traceback.format_exc()}"
            print(f"[UI] HTML 생성 예외: {error_msg}", file=sys.stderr)
            st.error(error_msg)
            st.stop()

    html_generation_end = time.time()
    timing_info["html_generation"] = html_generation_end - html_generation_start

    print("[UI] 스피너 종료, 세션 상태 저장 시작", file=sys.stderr)
    
    # 세션 상태에 결과 저장
    st.session_state.generated_html = html_content
    st.session_state.generated_image = saju_img
    st.session_state.html_filename = html_filename
    
    print("[UI] 세션 상태 저장 완료", file=sys.stderr)

    # 종료 시간 계산
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"[UI] 전체 프로세스 완료: {elapsed_time:.1f}초", file=sys.stderr)
    progress_log.success(f"✅ 6/6 단계 완료! 전체 소요 시간: {elapsed_time:.1f}초")

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

    # 스트리밍 표시를 위한 placeholder 생성
    chat_summary_placeholder = st.empty()
    with chat_summary_placeholder.container():
        st.markdown("#### 💬 채팅방 요약 (스트리밍 생성 중...)")
        streaming_text = st.empty()

    try:
        # 도사 스타일 요약 프롬프트 - {user_name} 치환
        chat_summary_instruction = locked_chat_summary_prompt.format(user_name=user_name)

        # 프롬프트에서 글자수 정보 추출 (없으면 기본값 사용)
        import re
        char_limit_match = re.search(r'(\d+)자\s*내외로?\s*요약.*?(?:최대\s*(\d+)자)?', locked_chat_summary_prompt)
        if char_limit_match:
            target_chars = int(char_limit_match.group(1))
            max_chars = int(char_limit_match.group(2)) if char_limit_match.group(2) else target_chars + 500
        else:
            target_chars = 2500
            max_chars = 3000

        chat_summary_msg = f"""다음은 {user_name}의 사주 내용입니다. 이를 도사 말투로 {target_chars}자 내외로 요약해주세요:

{full_text}

[요구사항]
- 도사 말투 사용
- {user_name}을(를) 호칭으로 사용
- 핵심 내용 포함: 총운, 연애운, 건강운, 직장운, 재물운, 월별운, 대길대흉 등
- {target_chars}자 내외 (최대 {max_chars}자)
- 밝고 유쾌하면서도 무게감 있게"""

        # 스트리밍 모드로 OpenAI API 호출
        chat_summary_stream = locked_openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": chat_summary_instruction},
                {"role": "user", "content": chat_summary_msg},
            ],
            stream=True
        )

        chat_summary_text = ""
        for chunk in chat_summary_stream:
            if chunk.choices[0].delta.content is not None:
                chat_summary_text += chunk.choices[0].delta.content
                # 실시간으로 텍스트 업데이트
                streaming_text.markdown(f"```\n{chat_summary_text}\n```")

        chat_summary_text = chat_summary_text.strip()

        # 세션 상태에 채팅방 요약 저장
        st.session_state["chat_summary"] = chat_summary_text

        # 스트리밍 완료 후 최종 결과 표시
        chat_summary_placeholder.empty()
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
