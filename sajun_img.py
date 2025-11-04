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

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

load_dotenv()

st.set_page_config(page_title="사주 → 이미지 생성기", page_icon="🧧", layout="wide")

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
OPENAI_TEXT_MODEL = "gpt-4.1"
OPENAI_IMAGE_MODEL = "gpt-image-1"
OPENAI_IMAGE_SIZE = "1024x1024"
DEFAULT_SYSTEM_INSTRUCTION = (
    "A mystical, hopeful scene rooted in Korean culture. "
    "Draw the characters in a way that highlights their personality, similar to Disney's Tangled and Encanto. "
    "The overall scene should be bright, rich in color, and vibrant, with a lovely emphasis on the characters. "
    "Express the faces in a Ghibli style. The lighting should be soft but powerful, and the characters should embody both warmth and vitality. "
    "The atmosphere should be both fantastical and dramatic."
)
DEFAULT_SUMMARY_INSTRUCTION = (
    "You are a Korean-to-English creative synthesis assistant with a warm, hopeful tone. "
    "Read the provided Korean saju text and create a vivid, single-scene description that can be rendered as one beautiful painting. "
    "Your description MUST include: "
    "1. WHO: A specific human figure (describe gender, youthful for their age, beautiful, and elegant appearance, attire, posture, must have no wrinkles) "
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

# ----------------------------
# UI
# ----------------------------
st.title("🧧 사주 텍스트 → 이미지 생성기")
st.caption("프롬프트 생성: gemini-2.0-flash · 이미지 생성: gemini-2.5-flash-image-preview (OpenAI 옵션 지원)")

gemini_client = get_gemini_client()
if GEMINI_API_KEY and not gemini_client:
    st.error("Google genai 클라이언트 초기화에 실패했습니다. GEMINI_API_KEY를 확인해주세요.")
    st.stop()

openai_client = get_openai_client()
openai_available = bool(openai_client)

# 디버깅 정보
st.write(f"DEBUG - OPENAI_API_KEY 설정 여부: {bool(OPENAI_API_KEY)}")
st.write(f"DEBUG - OpenAI 패키지 import 여부: {OpenAI is not None}")
st.write(f"DEBUG - openai_client 초기화 여부: {openai_client is not None}")
st.write(f"DEBUG - openai_available: {openai_available}")

if not gemini_client and not openai_available:
    st.error("사용 가능한 모델이 없습니다. GEMINI_API_KEY 또는 OPENAI_API_KEY를 설정해주세요.")
    st.stop()

if "core_scene_summary" not in st.session_state:
    st.session_state.core_scene_summary = ""

if not GEMINI_API_KEY:
    st.info("GEMINI_API_KEY가 설정되지 않아 OpenAI 옵션만 사용할 수 있습니다.")
if not openai_available:
    st.info("OPENAI_API_KEY가 설정되지 않았거나 openai 패키지가 없어 Google Gemini 옵션만 사용할 수 있습니다.")

colA, colB = st.columns(2)
with colA:
    saju_meongsik = st.text_area("사주명식", height=180, placeholder="예) 갑자 庚子年 丙寅月 壬午日 乙巳時 ...")
with colB:
    saju_puli = st.text_area("사주풀이", height=180, placeholder="예) 화(火) 기운이 강하고 금/수 보완이 필요… 등 해석 요약")

system_prompt_input = st.text_area(
    "이미지 생성 시스템 프롬프트",
    value=DEFAULT_SYSTEM_INSTRUCTION,
    height=180,
    help="이미지 프롬프트 작성 모델에 전달할 시스템 메시지입니다.",
)
system_prompt = system_prompt_input if system_prompt_input.strip() else DEFAULT_SYSTEM_INSTRUCTION

summary_prompt_input = st.text_area(
    "장면요약 시스템 프롬프트",
    value=DEFAULT_SUMMARY_INSTRUCTION,
    height=180,
    help="핵심 장면 요약 생성 모델에 전달할 시스템 메시지입니다.",
)
summary_prompt = summary_prompt_input if summary_prompt_input.strip() else DEFAULT_SUMMARY_INSTRUCTION

model_col_1, model_col_2 = st.columns(2)
prompt_options = []
image_options = []
if gemini_client:
    prompt_options.append(("Google Gemini (gemini-2.5-pro)", ("gemini", "gemini-2.5-pro")))
    prompt_options.append(("Google Gemini (gemini-2.5-flash)", ("gemini", "gemini-2.5-flash")))
    image_options.append(("Google Gemini (gemini-2.5-flash-image-preview)", "gemini"))
if openai_available:
    prompt_options.append(("OpenAI GPT-4.1", ("openai", "gpt-4.1")))
    prompt_options.append(("OpenAI GPT-4.1-Mini", ("openai", "gpt-4.1-mini")))
    image_options.append(("OpenAI GPT-Image-1", "openai"))

with model_col_1:
    prompt_model_choice = st.selectbox(
        "프롬프트 모델 선택",
        options=[label for label, _ in prompt_options],
        index=0,
    )

prompt_provider, openai_text_model_selected = dict(prompt_options)[prompt_model_choice]
if prompt_provider != "openai":
    openai_text_model_selected = OPENAI_TEXT_MODEL

with model_col_2:
    image_model_choice = st.selectbox(
        "이미지 모델 선택",
        options=[label for label, _ in image_options],
        index=0,
    )

image_provider = dict(image_options)[image_model_choice]

mode = st.radio(
    "생성 기준 선택",
    ("사주명식으로 이미지 만들기", "사주풀이로 이미지 만들기"),
    horizontal=True
)

st.markdown("---")
generate = st.button("🚀 이미지 생성", type="primary", use_container_width=True)

if generate:
    if not saju_meongsik and not saju_puli:
        st.error("사주명식 또는 사주풀이 중 하나 이상을 입력해주세요.")
        st.stop()

    base_text = saju_meongsik if mode == "사주명식으로 이미지 만들기" else saju_puli
    if not base_text.strip():
        st.error("선택한 항목의 텍스트가 비어 있습니다.")
        st.stop()

    with st.spinner("🔍 핵심 장면 추출 중..."):
        try:
            core_scene = summarize_for_visuals(
                base_text,
                provider=prompt_provider,
                gemini_client=gemini_client,
                openai_client=openai_client,
                system_instruction=summary_prompt,
                openai_text_model=openai_text_model_selected,
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
                system_instruction=system_prompt,
                provider=prompt_provider,
                gemini_client=gemini_client,
                openai_client=openai_client,
                core_scene=core_scene,
                openai_text_model=openai_text_model_selected,
            )
        except Exception as exc:
            st.error(f"프롬프트 생성 중 오류가 발생했습니다: {exc}")
            st.stop()

    if not prompt:
        st.error("프롬프트 생성에 실패했습니다. 입력 내용을 다시 확인해주세요.")
        st.stop()

    final_prompt = prompt

    with st.spinner("🎨 이미지 생성 중..."):
        imgs = generate_images(
            final_prompt,
            num_images=1,
            provider=image_provider,
            gemini_client=gemini_client,
            openai_client=openai_client,
        )

    valid = [i for i in imgs if i is not None]
    if valid:
        st.success(f"✅ 이미지 {len(valid)}장 생성 완료!")
        st.markdown("### 생성 결과")
        cols = st.columns(3)
        for idx, im in enumerate(imgs):
            if im is None: 
                continue
            with cols[idx % 3]:
                st.image(im, caption=f"생성 이미지 #{idx+1}", use_container_width=True)
                buf = BytesIO()
                im.save(buf, format="PNG")
                st.download_button(
                    label=f"📥 다운로드 #{idx+1}",
                    data=buf.getvalue(),
                    file_name=f"saju_generated_{int(time.time())}_{idx+1}.png",
                    mime="image/png",
                    key=f"dl_{idx}"
                )
    else:
        st.warning("이미지 생성 결과가 비어 있습니다. 텍스트를 더 구체적으로 수정해 다시 시도해주세요.")

if not generate:
    summary_display = st.session_state.get("core_scene_summary", "").strip()
    if summary_display:
        st.markdown("#### ✨ 핵심 장면 요약")
        st.write(summary_display)
