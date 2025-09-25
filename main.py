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
load_dotenv()

st.set_page_config(page_title="AI 육아 도우미 봇", layout="centered")

try:
    HIRA_API_SERVICE_KEY = os.environ.get("HIRA_API_SERVICE_KEY") or st.secrets.get("HIRA_API_SERVICE_KEY")
    KAKAO_API_REST_KEY = os.environ.get("KAKAO_API_REST_KEY") or st.secrets.get("KAKAO_API_REST_KEY")
    if not HIRA_API_SERVICE_KEY or not KAKAO_API_REST_KEY:
        raise KeyError
except KeyError:
    st.error("필수 API 키가 설정되지 않았습니다. .env 파일 또는 Streamlit secrets를 확인해주세요.")
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
    
    try:
        index = faiss.read_index(LOCAL_INDEX_PATH)
        with open(LOCAL_META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except FileNotFoundError as e:
        st.error(f"필수 데이터 파일을 찾을 수 없습니다: {e}. 'faiss.index'와 'faiss.meta.json' 파일이 있는지 확인하세요.")
        st.stop()
        
    return embed_model, tokenizer, model, index, meta["texts"], meta.get("metas")

with st.spinner("AI 모델과 육아 데이터를 불러오는 중입니다... (최초 실행 시 시간이 다소 걸릴 수 있습니다)"):
    embed_model, tokenizer, model, index, TEXTS, METAS = load_all_models_and_data()


# --- 3. 핵심 기능 함수 정의 ---

def generate_text(prompt_text, max_tokens, temperature=0.1):
    """LLM을 직접 호출하여 텍스트를 생성하는 통합 함수 (의도분석용)"""
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
            json_part_start = full_response.find('{')
            json_part_end = full_response.rfind('}') + 1
            if json_part_start != -1 and json_part_end != 0:
                json_part = full_response[json_part_start:json_part_end]
                intent = json.loads(json_part)
                intent['action'] = 'search'
                return intent
            else:
                 return {"action": "chat"}
        except (json.JSONDecodeError, Exception):
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
        
        sorted_items = sorted(items, key=lambda x: float(x.get('distance', 99999)))
        today_weekday = datetime.now().weekday() + 1
        
        response_text = f"✅ **'{place_type}'** 검색 결과입니다. (거리 순, 최대 5곳)\n\n"
        for item in sorted_items[:5]:
            response_text += f"**{item['yadmNm']}**\n- 주소: {item['addr']}\n"
            telno = item.get('telno', '정보 없음')
            tel_link = ''.join(filter(str.isdigit, telno)) if telno != '정보 없음' else ''
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


SYSTEM_PROMPT_RAG = """
# 역할 정의
당신은 대한민국 부모들을 위한 **육아 전문 어시스턴트 챗봇**입니다. 당신의 목표는 부모들의 육아 관련 질문에 대해 가장 신뢰할 수 있고 실용적인 정보를 제공하는 것입니다. 항상 따뜻하고, 공감적이며, 격려하는 태도를 유지하세요.

# 답변 생성 프로세스
당신은 다음의 논리적 순서에 따라 답변 모드를 결정하고 생성해야 합니다.

1.  **[CASE C 확인]:** 사용자의 질문이 이전 대화에 대한 단순 확인, 감사, 짧은 의견 등인지 먼저 확인합니다. 만약 그렇다면, `[C. 짧은 대화형 답변]` 규칙에 따라 자연스럽게 대답하고 프로세스를 종료합니다.
2.  **[CASE A 확인]:** 질문이 정보 제공을 요구하며 '참고 자료'가 있는 경우, `[A. RAG 기반 답변 형식]`을 반드시 준수하여 답변을 생성합니다.
3.  **[CASE B 적용]:** 위 두 경우에 해당하지 않으면, `[B. 일반 답변 형식]`에 따라 당신의 지식으로 답변을 생성합니다.

---

# [A. RAG 기반 답변 형식]
'참고 자료'가 있을 때 사용하는 엄격한 출력 형식입니다. 아래 4개 섹션을 **정확히** 지켜서 출력해야 합니다.

1.  **✅ 핵심 요약:**
    * 질문에 대한 가장 중요한 답변을 3문장 이내로 요약합니다.
2.  **👶 단계별 가이드:**
    * 부모가 실제로 따라 할 수 있는 구체적인 행동 지침을 3~10개의 번호 목록으로 제시합니다. 각 항목은 중복된 문장이 없게 한 문장으로 간결하게 작성합니다.
3.  **📌 근거:**
    * 답변의 근거가 된 '참고 자료'의 출처(`id / title / category`)를 명시합니다. 동일한 출처는 한 번만 언급합니다.
4.  **⚠️ 주의/면책:**
    * 의학적 조언의 한계를 명시하고, 전문가 상담의 중요성을 2문장 이내로 강조합니다. 매번 다른 문장으로 표현하세요.

---

# [B. 일반 답변 형식]
'참고 자료'가 없을 때 사용하는 유연한 출력 형식입니다.

1.  **공감적 도입:**
    * "많이 힘드시겠어요.", "많은 부모님들이 겪는 고민이에요." 등 사용자의 상황에 공감하는 문장으로 시작하세요.
2.  **체계적인 조언:**
    * **소제목**, **글머리 기호(•)**, **이모지(💡, 🧸 등)**를 자유롭게 활용하여 실용적인 조언을 명확하고 읽기 쉽게 제시하세요. 정보의 나열이 아닌, 실제 도움이 되는 팁 중심으로 구성합니다.
3.  **전문가 조언 권장:**
    * 답변 마지막에는 "아이의 상태가 지속되거나 걱정되신다면, 소아과 전문의와 상담해보시는 것이 가장 정확합니다."와 같이 전문가 상담을 권장하는 내용을 부드럽게 포함하여 책임을 명확히 하세요.

---

# [C. 짧은 대화형 답변]
사용자의 질문이 아래와 같을 경우, A나 B 형식을 사용하지 않고 자연스러운 일상 대화처럼 한두 문장으로 짧고 간결하게 답변합니다.
- 이전 대화 내용에 대한 단순 확인 (예: "방금 몇 도라고 했지?", "아까 말한 첫번째 방법이 뭐였어?")
- 감사 표현 (예: "고마워", "도움이 됐어")
- 간단한 인사나 짧은 의견 (예: "그렇구나", "알겠어")

---

# 공통 규칙 및 제약사항 (모든 답변에 적용)
* **어조:** 항상 친절하고, 긍정적이며, 부모를 지지하는 격려의 말투를 사용하세요.
* **의료적 조언 금지:** 당신은 의사가 아닙니다. 확정적인 질병을 진단하거나 특정 약물을 추천하는 등 의료 전문가의 영역을 절대 침범하지 마세요.
* **응급 상황:** 사용자의 질문 내용에 고열, 호흡 곤란, 의식 저하, 경련, 심한 탈수 등 **응급 징후**가 포함되어 있다면, 다른 정보보다 **즉시 119 신고 또는 응급실 방문**을 최우선으로 안내하세요.
* **언어:** 반드시 명확하고 이해하기 쉬운 한국어로만 답변합니다. 전문 용어 사용을 최소화하세요.
"""

def retrieve(query, top_k=5):
    q_emb = embed_model.encode([query], normalize_embeddings=True).astype("float32")
    _, indices = index.search(q_emb, top_k)
    return [{"meta": METAS[i], "text": TEXTS[i]} for i in indices[0]]

def build_context(docs):
    context = []
    # 중복된 출처를 제거하기 위한 set
    seen_ids = set()
    for doc in docs:
        meta = doc.get("meta", {})
        doc_id = meta.get('id', 'N/A')
        # id가 없거나 이미 본 id인 경우 건너뛰기
        if doc_id == 'N/A' or doc_id in seen_ids:
            continue
        context.append(f"출처: {doc_id} | 제목: {meta.get('title', 'N/A')}\n{doc.get('text', '')}")
        seen_ids.add(doc_id)
    return "\n\n".join(context)

def ask_rag(conversation_history):
    """
    [수정된 버전]
    순수한 대화 기록과 RAG 컨텍스트를 분리하여 LLM에 명확한 지시를 내립니다.
    """
    # RAG 검색을 위해 최신 사용자 질문 추출
    user_prompt = conversation_history[-1]["content"]
    docs = retrieve(user_prompt)
    context_str = build_context(docs)

    # [수정] LLM에 전달할 메시지 리스트를 새로 구성
    # 1. 시스템 프롬프트
    # 2. 순수한 이전 대화 기록
    # 3. 마지막에 RAG 컨텍스트와 새로운 질문을 함께 담은 명확한 지시문
    messages_for_llm = [
        {"role": "system", "content": SYSTEM_PROMPT_RAG}
    ]

    # 이전 대화 기록을 그대로 추가
    # 마지막 사용자 질문은 제외하고 추가 (나중에 포맷팅해서 넣을 것이므로)
    messages_for_llm.extend(conversation_history[:-1])

    # [변경] RAG 컨텍스트와 최신 질문을 결합한 명확한 지시 프롬프트 생성
    # 참고 자료가 비어있는 경우(RAG 검색 결과가 없는 경우)를 대비
    if context_str.strip():
        augmented_prompt = (
            f"아래 '참고 자료'와 이전 대화 내용을 종합적으로 고려하여 다음 질문에 답변해줘.\n\n"
            f"--- 참고 자료 ---\n{context_str}\n------------------\n\n"
            f"질문: {user_prompt}"
        )
    else:
        # RAG 검색 결과가 없으면, 이전 대화 내용만 바탕으로 답변하도록 요청
        augmented_prompt = (
            f"이전 대화 내용을 바탕으로 다음 질문에 답변해줘.\n\n"
            f"질문: {user_prompt}"
        )

    # 마지막 user 메시지로 지시 프롬프트를 추가
    messages_for_llm.append({"role": "user", "content": augmented_prompt})

    # 최종 프롬프트 생성 및 LLM 호출
    prompt = tokenizer.apply_chat_template(messages_for_llm, tokenize=False, add_generation_prompt=True)
    
    raw_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**raw_inputs, max_new_tokens=1024, do_sample=True, temperature=0.2, top_p=0.95)
    
    # 프롬프트를 제외한 순수 답변만 추출
    response_only = tokenizer.decode(outputs[0][raw_inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
    
    return response_only

# --- 4. Streamlit UI 및 메인 로직 ---
st.title("AI 육아 도우미 봇 🤖")
st.info("임신, 출산, 육아 질문은 물론, '부천 원종동 주변 소아과 찾아줘'처럼 장소 검색도 가능해요!")

st.sidebar.title("메뉴")
st.sidebar.info("이곳에 추가 정보나 링크를 넣을 수 있습니다.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("질문을 입력해주세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant"):
        with st.spinner("AI가 열심히 생각하고 있어요... 🤔"):
            user_prompt = st.session_state.messages[-1]["content"] # 최신 프롬프트는 의도 파악에 사용
            intent = parse_intent_with_ai(user_prompt)
            action = intent.get("action", "chat")

            if action == "search":
                response_content = handle_search(intent)
            else:
                response_content = ask_rag(st.session_state.messages)

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

/* === 히어로 이미지 위치(좌상/우하) === */
.hero-left, .hero-right{
  position:fixed; max-width:none !important; height:auto; object-fit:contain;
  border-radius:12px; box-shadow:none !important; border:none !important; background:transparent !important; pointer-events:none; z-index:0;
}
.hero-left{ top:120px; left:48px; width:220px; }
.hero-right{ top:360px; right:48px; width:240px; }


@media (max-width: 900px){ .hero-left, .hero-right{ display:none; } }
@media (max-width: 640px){ [data-testid="stImage"] img{ max-height:32vh; } }
</style>
""", unsafe_allow_html=True)

def image_to_data_uri(path):
    if not os.path.exists(path):
        return ""
    try:
        import base64
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        return f"data:image/png;base64,{encoded}"
    except Exception:
        return ""

left_uri = image_to_data_uri("1.png")
right_uri = image_to_data_uri("2.png")
if left_uri:
    st.markdown(f'<img class="hero-left" src="{left_uri}" alt="left decoration">', unsafe_allow_html=True)
if right_uri:
    st.markdown(f'<img class="hero-right" src="{right_uri}" alt="right decoration">', unsafe_allow_html=True)
