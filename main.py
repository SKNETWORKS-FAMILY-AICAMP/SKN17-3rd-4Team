import streamlit as st
import json
import os
import requests
import torch
import faiss
import numpy as np
from dotenv import load_dotenv
from urllib.parse import unquote
from datetime import datetime
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- 1. 초기 설정 ---

# .env 파일에서 환경 변수 로드
load_dotenv()

# Streamlit 페이지 설정
st.set_page_config(page_title="AI 육아 도우미 봇", layout="centered")

# API 키 로드
try:
    HIRA_API_SERVICE_KEY = os.environ["HIRA_API_SERVICE_KEY"]
    KAKAO_API_REST_KEY = os.environ["KAKAO_API_REST_KEY"]
except KeyError as e:
    st.error(f"필수 API 키가 .env 파일에 설정되지 않았습니다: {e}")
    st.stop()

# --- 2. 모델 및 데이터 로딩 ---
@st.cache_resource
def load_all_models_and_data():
    """RAG에 필요한 모든 모델과 데이터를 로드합니다."""
    HF_EMBED_REPO_ID = "WOOJINIYA/parentcare-bot-bge-m3"
    HF_LLM_REPO_ID = "WOOJINIYA/parentcare-bot-qwen2.5-7b"
    LOCAL_INDEX_PATH = "faiss.index"
    LOCAL_META_PATH = "faiss.meta.json"

    embed_model = SentenceTransformer(HF_EMBED_REPO_ID)
    tokenizer = AutoTokenizer.from_pretrained(HF_LLM_REPO_ID)
    model = AutoModelForCausalLM.from_pretrained(
        HF_LLM_REPO_ID, device_map="auto", torch_dtype=torch.float16, load_in_4bit=True
    )
    index = faiss.read_index(LOCAL_INDEX_PATH)
    with open(LOCAL_META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return embed_model, tokenizer, model, index, meta["texts"], meta.get("metas")

with st.spinner("AI 모델과 육아 데이터를 불러오는 중입니다... (최초 실행 시 시간이 다소 걸릴 수 있습니다)"):
    embed_model, tokenizer, model, index, TEXTS, METAS = load_all_models_and_data()


# --- 3. 핵심 기능 함수 정의 ---

def generate_text(prompt_text, max_tokens, temperature=0.1):
    """LLM을 직접 호출하여 텍스트를 생성하는 통합 함수"""
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    outputs = model.generate(
        **inputs, max_new_tokens=max_tokens, do_sample=True, temperature=temperature, top_p=0.95
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def parse_intent_with_ai(user_prompt):
    """사용자의 의도를 'search' 또는 'chat'으로 분류하고, 관련 정보를 추출합니다."""
    search_keywords = ['주변', '찾아줘', '어디', '소아과', '산부인과', '약국', '병원']
    if any(keyword in user_prompt for keyword in search_keywords):
        system_prompt = "사용자의 메시지에서 'place'와 'location'을 추출하여 JSON 형식으로만 답변해라. 'place'는 병원, 약국, 소아과, 산부인과 등 장소 종류를 의미하고 'location'은 '서울 대방동' 같은 지역명을 의미한다. 둘 중 하나라도 없으면 null로 표시해라. 예시: {\"place\": \"소아과\", \"location\": \"서울 대방동\"}"
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full_response = generate_text(prompt, 128)
        try:
            json_part = full_response[full_response.find('{'):full_response.rfind('}')+1]
            intent = json.loads(json_part)
            intent['action'] = 'search'
            return intent
        except Exception:
            return {"action": "chat"}
    return {"action": "chat"}

# --- 장소 검색 (Kakao & HIRA API) 관련 함수 ---
def geocode_kakao(address):
    endpoint = "https://dapi.kakao.com/v2/local/search/address.json"
    headers = {"Authorization": f"KakaoAK {KAKAO_API_REST_KEY}"}
    params = {'query': address}
    try:
        response = requests.get(endpoint, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data['documents']:
            doc = data['documents'][0]
            return float(doc['y']), float(doc['x']), None
        return None, None, f"'{address}'에 대한 검색 결과가 없습니다."
    except Exception as e:
        return None, None, f"카카오맵 API 오류: {e}"

def _search_hira(lat, lon, place_type, endpoint):
    params = {"serviceKey": unquote(HIRA_API_SERVICE_KEY), "xPos": lon, "yPos": lat, "radius": 3000, "_type": "json"}
    try:
        response = requests.get(endpoint, params=params, timeout=10)
        response.raise_for_status()
        items = response.json().get('response', {}).get('body', {}).get('items', {}).get('item', [])
        if not items: return f"주변 3km 이내에 검색된 '{place_type}'이(가) 없습니다."
        
        sorted_items = sorted(items, key=lambda x: float(x['distance']))
        today_weekday = datetime.now().weekday() + 1
        
        response_text = f"✅ **'{place_type}'** 검색 결과입니다. (거리 순, 최대 5곳)\n\n"
        for item in sorted_items[:5]:
            response_text += f"**{item['yadmNm']}**\n- 주소: {item['addr']}\n"
            telno = item.get('telno', '정보 없음')
            tel_link = ''.join(filter(str.isdigit, telno)) if telno else ''
            response_text += f"- 전화: [{telno}](tel:{tel_link})\n"
            start_time, close_time = item.get(f'dutyTime{today_weekday}s'), item.get(f'dutyTime{today_weekday}c')
            operating_hours = f"{start_time[:2]}:{start_time[2:]} ~ {close_time[:2]}:{close_time[2:]}" if start_time and close_time else "정보 없음"
            response_text += f"- 오늘 영업시간: {operating_hours} (방문 전 확인 필수)\n---\n"
        return response_text
    except Exception as e:
        return f"{place_type} API 요청 중 오류가 발생했습니다: {e}"

def search_hospitals(lat, lon, place_type):
    return _search_hira(lat, lon, place_type, "http://apis.data.go.kr/B551182/hospInfoServicev2/getHospBasisList")
def search_pharmacies(lat, lon):
    return _search_hira(lat, lon, "약국", "http://apis.data.go.kr/B551182/pharmacyInfoService/getParmacyBasisList")

def handle_search(intent):
    location_name = intent.get("location")
    place_type = intent.get("place", "병원")
    if not location_name: return "어디 주변에서 찾아드릴까요? '서울 대방동'처럼 지역 이름을 알려주세요."
    lat, lon, error_message = geocode_kakao(location_name)
    if error_message: return f"⚠️ 위치 변환 중 오류가 발생했습니다:\n\n`{error_message}`"
    if lat and lon:
        return search_pharmacies(lat, lon) if "약국" in place_type else search_hospitals(lat, lon, place_type)

# --- RAG 기반 육아 질문 답변 함수 ---
SYSTEM_PROMPT_RAG = """
당신은 한국 부모를 돕는 육아 도우미 챗봇입니다.

출력 형식(정확히 이 4개 섹션만, 각 섹션 규칙 엄수):
1) ✅ 핵심 요약:
      문장 3개 이내.

2) 👶 단계별 가이드:
      번호 리스트 3~10개. 각 항목은 한 문장.

3) 📌 근거:
      각 줄에 (id / title / category) 만약에 똑같은 출처가 있다면 1개씩만 출력해.

4) ⚠️ 주의/면책:
      문장 2개 이내. 같은 문장을 반복 금지.

규칙:
- 어떤 질병이 의심되면 질병명도 함께 제시한다.
- 의학/정책 정보는 반드시 근거(기관명+날짜+URL)를 함께 제시한다.
- 근거 불명확 시 '확인 필요'라고만 쓰고, 일반적 주의사항을 간결히 제시한다.
- 응급 징후(고열, 호흡곤란, 경련, 탈수 등) 언급 시 119/응급실 안내를 포함한다.
- 같은 문장/문구 반복 금지. 이미 말한 내용 재서술 금지.
- 위 4개 섹션 출력 후 즉시 종료(추가 문장 금지).
- 답변 형식 외에 어떤 부가적인 설명, 주석, 태그도 절대로 출력하지 않는다.
"""

def retrieve(query, top_k=5):
    q_emb = embed_model.encode([query], normalize_embeddings=True).astype("float32")
    _, indices = index.search(q_emb, top_k)
    return [{"meta": METAS[i], "text": TEXTS[i]} for i in indices[0]]

def build_context(docs):
    context = []
    for i, doc in enumerate(docs, 1):
        meta, text = doc["meta"], doc["text"]
        context.append(f"[{i}] 출처: {meta.get('id', 'N/A')} | 제목: {meta.get('title', 'N/A')}\n{text}")
    return "\n\n".join(context)

def ask_rag(user_prompt):
    docs = retrieve(user_prompt)
    context_str = build_context(docs)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_RAG},
        {"role": "user", "content": f"질문: {user_prompt}\n\n아래 참고 자료를 바탕으로 답변해줘.\n\n{context_str}"}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_response = generate_text(prompt, 512, temperature=0.2)
    return full_response[len(prompt)-len(" <|im_start|>assistant\n"):] # 프롬프트 부분 제외

# --- 4. Streamlit UI 및 메인 로직 ---
st.title("AI 육아 도우미 봇 🤖")
st.image("img.jpg", use_container_width=True)
st.info("임신, 출산, 육아 질문은 물론, '부천 원종동 주변 소아과 찾아줘'처럼 장소 검색도 가능해요!")

# 사이드바
st.sidebar.title("메뉴")
# ... (기존 사이드바 코드는 여기에 그대로 붙여넣기)

# 대화 기록 초기화 및 표시
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 및 챗봇 응답 처리 로직
if prompt := st.chat_input("질문을 입력해주세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    user_prompt = st.session_state.messages[-1]["content"]
    with st.chat_message("assistant"):
        with st.spinner("AI가 열심히 생각하고 있어요... 🤔"):
            intent = parse_intent_with_ai(user_prompt)
            action = intent.get("action", "chat")

            if action == "search":
                response_content = handle_search(intent)
            else: # action == "chat"
                response_content = ask_rag(user_prompt)

    st.session_state.messages.append({"role": "assistant", "content": response_content})
    st.rerun()


# --- CSS 스타일링 및 이미지 오버레이 코드 ---

st.markdown("""
<style>
/* === 기본 폰트/색 토큰 === */
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;600;700&display=swap');
:root{
  --brand:#6C7CFF;
  --bg:#F7F9FC;
  --surface:#FFFFFF;
  --line:#E9EEF5;
  --text:#111827;
  --radius:12px;
  --radius-lg:16px;
  --font:'Noto Sans KR', system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
}
@media (prefers-color-scheme: dark){
  :root{ --bg:#0B1220; --surface:#0F1629; --line:#1E2A44; --text:#E5E7EB; }
}
body, div, p, span, h1, h2, h3, h4, h5, h6{ font-family:var(--font); }

/* === 배경/레이아웃 최소 === */
.main{
  background:
    radial-gradient(1200px 600px at 10% -10%, rgba(108,124,255,.06) 0%, rgba(108,124,255,0) 50%),
    linear-gradient(180deg, var(--bg) 0%, var(--surface) 64%);
}
section.main > div.block-container{ padding-top:1rem; padding-bottom:1.4rem; }

/* 모션 최소화 존중 */
@media (prefers-reduced-motion: reduce){ *{animation:none !important; transition:none !important;} }

/* === 타이틀만 가볍게 === */
.block-container h1:first-of-type{
  margin-bottom:.35rem; font-weight:800; letter-spacing:-.2px;
  background:linear-gradient(90deg, var(--text), color-mix(in oklab, var(--text), #334155 60%));
  -webkit-background-clip:text; background-clip:text; color:transparent;
}

/* === 이미지 공통(그림자/테두리 제거) === */
[data-testid="stImage"] img{
  display:block; width:100%; height:auto; max-height:42vh; object-fit:cover;
  border-radius:var(--radius-lg); box-shadow:none !important; border:none !important; background:transparent !important;
}

/* === 채팅 입력창 최소 === */
[data-testid="stChatInput"]{ border-top:1px dashed var(--line); padding-top:.6rem; margin-top:.6rem; }
[data-testid="stChatInput"] textarea{
  border:1px solid var(--line) !important; border-radius:var(--radius) !important; box-shadow:0 2px 10px rgba(0,0,0,.03) !important;
}

/* === 워터마크(채팅창 위 고정) === */
.ux-footer{ color:#6B7280; font-size:12px; text-align:center; margin-top:14px; user-select:none; }
.ux-over-chat{
  position:fixed; left:0; right:0; bottom:72px; padding:.25rem 0; background:transparent; z-index:5; pointer-events:none;
}
@media (max-width: 640px){ .ux-over-chat{ bottom:88px; } }

/* === 히어로 이미지 위치(좌상/우하) === */
.hero-left, .hero-right{
  position:fixed; max-width:none !important; height:auto; object-fit:contain;
  border-radius:12px; box-shadow:none !important; border:none !important; background:transparent !important; pointer-events:none; z-index:0;
}
.hero-left{ top:120px; left:48px; width:220px; }   /* 화면 위쪽 왼쪽 */
.hero-right{ top:360px; right:48px; width:240px; } /* 화면 오른쪽 아래 */


@media (max-width: 900px){ .hero-left, .hero-right{ display:none; } }

/* === 작은 화면 보정 === */
@media (max-width: 640px){ [data-testid="stImage"] img{ max-height:32vh; } }
</style>
""", unsafe_allow_html=True)

import base64 as _b64
def _img_to_data_uri(path):
    try:
        with open(path, "rb") as _f:
            return "data:image/png;base64," + _b64.b64encode(_f.read()).decode("utf-8")
    except Exception:
        return ""
_left_uri = _img_to_data_uri("1.png")
_right_uri = _img_to_data_uri("2.png")
if _left_uri:
    st.markdown(f'<img class="hero-left" src="{_left_uri}" alt="left">', unsafe_allow_html=True)
if _right_uri:
    st.markdown(f'<img class="hero-right" src="{_right_uri}" alt="right">', unsafe_allow_html=True)