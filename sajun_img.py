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
    import openai
except ImportError:
    openai = None

load_dotenv()

st.set_page_config(page_title="사주 → 이미지 생성기", page_icon="🧧", layout="wide")

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
    "You are an expert visual prompt writer for an image generation model. "
    "Read the provided Korean saju text and produce ONE concise, high-quality image prompt suitable for "
    "photorealistic or illustrative rendering. Avoid fortune-telling claims; instead focus on visual metaphors, "
    # 👇 수정된 부분
    "**which may include a human figure used symbolically or metaphorically (e.g., their posture, attire, or interaction with elements)**, "
    "colors, elements (wood/fire/earth/metal/water), seasonal motifs, props, and composition. "
    # 👆 수정된 부분
    "Include style, lighting, mood, and camera/composition details. "
    "Return ONLY the final prompt in English."
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

def write_prompt_from_saju(
    source_text: str,
    system_instruction: str = DEFAULT_SYSTEM_INSTRUCTION,
    provider: str = "gemini",
    gemini_client: Optional[genai.Client] = None,
) -> str:
    """
    사주 텍스트(명식/풀이) -> 이미지용 프롬프트 1개 생성
    """
    user_msg = f"""
[SAJU TEXT / Korean]
{source_text}

[REQUEST]
- Compose one scene that visually symbolizes the above text
- Include subject, background, props, color palette, texture, lighting, mood, and art style
- Prefer 16:9 composition; high fidelity wording (but avoid exaggerated numeric buzzwords)
"""
    if provider == "openai":
        if not openai or not OPENAI_API_KEY:
            raise ValueError("OpenAI 지원이 활성화되지 않았습니다.")
        openai.api_key = OPENAI_API_KEY
        completion = openai.chat.completions.create(
            model=OPENAI_TEXT_MODEL,
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

def generate_images(
    prompt: str,
    num_images: int = 3,
    provider: str = "gemini",
    gemini_client: Optional[genai.Client] = None,
):
    """
    텍스트만으로 이미지 생성. 최대 num_images장 시도.
    반환: PIL.Image 또는 None의 리스트
    """
    images = []
    if provider == "openai":
        if not openai or not OPENAI_API_KEY:
            return [None] * num_images
        openai.api_key = OPENAI_API_KEY
        for _ in range(num_images):
            try:
                response = openai.images.generate(
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

openai_available = bool(openai and OPENAI_API_KEY)
if not gemini_client and not openai_available:
    st.error("사용 가능한 모델이 없습니다. GEMINI_API_KEY 또는 OPENAI_API_KEY를 설정해주세요.")
    st.stop()

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
    "시스템 프롬프트",
    value=DEFAULT_SYSTEM_INSTRUCTION,
    height=180,
    help="프롬프트 작성 모델에 전달할 시스템 메시지입니다.",
)
system_prompt = system_prompt_input if system_prompt_input.strip() else DEFAULT_SYSTEM_INSTRUCTION

model_col_1, model_col_2 = st.columns(2)
prompt_options = []
image_options = []
if gemini_client:
    prompt_options.append(("Google Gemini (gemini-2.5-pro)", "gemini"))
    image_options.append(("Google Gemini (gemini-2.5-flash-image-preview)", "gemini"))
if openai_available:
    prompt_options.append(("OpenAI GPT-4.1", "openai"))
    image_options.append(("OpenAI GPT-Image-1", "openai"))

with model_col_1:
    prompt_model_choice = st.selectbox(
        "프롬프트 모델 선택",
        options=[label for label, _ in prompt_options],
        index=0,
    )
with model_col_2:
    image_model_choice = st.selectbox(
        "이미지 모델 선택",
        options=[label for label, _ in image_options],
        index=0,
    )

prompt_provider = dict(prompt_options)[prompt_model_choice]
image_provider = dict(image_options)[image_model_choice]

mode = st.radio(
    "생성 기준 선택",
    ("사주명식으로 이미지 만들기", "사주풀이로 이미지 만들기"),
    horizontal=True
)

quality_boost = st.checkbox("품질 강화(상세 묘사/전문 아트/16:9)", value=True)

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

    with st.spinner("📝 프롬프트 작성 중..."):
        try:
            prompt = write_prompt_from_saju(
                base_text,
                system_instruction=system_prompt,
                provider=prompt_provider,
                gemini_client=gemini_client,
            )
        except Exception as exc:
            st.error(f"프롬프트 생성 중 오류가 발생했습니다: {exc}")
            st.stop()

    if not prompt:
        st.error("프롬프트 생성에 실패했습니다. 입력 내용을 다시 확인해주세요.")
        st.stop()

    final_prompt = prompt
    if quality_boost:
        final_prompt += ", highly detailed, professional artwork, 16:9 composition"

    with st.spinner("🎨 이미지 생성 중..."):
        imgs = generate_images(
            final_prompt,
            num_images=3,
            provider=image_provider,
            gemini_client=gemini_client,
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
                st.image(im, caption=f"생성 이미지 #{idx+1}", use_column_width=True)
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
