# 육아 논문 데이터 기반 sLLM - RAG 프로젝트

본 프로젝트는 육아·보육 관련 내외부 문서를 기반으로 한 질의응답 시스템을 구축하는 것을 목표로 합니다.<br>
LLM에 직접 파인튜닝을 적용하지 않고, Retrieval-Augmented Generation(RAG) 방식을 통해 문서 검색 결과를 컨텍스트로 활용하여 신뢰성 있는 답변을 제공합니다.

<br>

## 1. 👨‍👩‍👧‍👦 팀 소개

<h2>육아 복지부</h2>

<table align="center">
  <tr>
    <td align="center" valign="top" style="padding: 10px;">
      <strong>김민균</strong><br/>
      <img src="https://github.com/user-attachments/assets/b242f6f7-423a-441f-9fed-65e754f4aa93" width="150" alt="김민균"/>
    </td>
    <td align="center" valign="top" style="padding: 10px;">
      <strong>김세한</strong><br/>
      <img src="https://github.com/user-attachments/assets/565cf252-2433-4bcb-9c82-1bbd35e42d8a" width="150" alt="김세한"/>
    </td>
    <td align="center" valign="top" style="padding: 10px;">
      <strong>김수현</strong><br/>
      <img src="https://github.com/user-attachments/assets/b3101204-db35-48ed-823c-66e9e441ccba" width="150" alt="김수현"/>
    </td>
    <td align="center" valign="top" style="padding: 10px;">
      <strong>정의중</strong><br/>
      <img src="https://github.com/user-attachments/assets/c0790bf8-cc79-4e38-b0d1-b49a27eadbab" width="150" alt="정의중"/>
    </td>
    <td align="center" valign="top" style="padding: 10px;">
      <strong>최우진</strong><br/>
      <img src="https://github.com/user-attachments/assets/d8451ecc-a69e-46ec-b00f-dbd2eefec6e0" width="150" alt="최우진"/>
    </td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/alswhitetiger">@alswhitetiger</a></td>
    <td align="center"><a href="https://github.com/kimsehan11">@kimsehan11</a></td>
    <td align="center"><a href="https://github.com/K-SH98">@K-SH98</a></td>
    <td align="center"><a href="https://github.com/uii42">@uii42</a></td>
    <td align="center"><a href="https://github.com/CHUH00">@CHUH00</a></td>
  </tr>
</table>

<br><br/>

<br>

## 2. 프로젝트 개요

- **프로젝트 소개**: 본 프로젝트는 넘쳐나는 육아 정보 속에서 신뢰할 수 있는 정보를 찾기 어려운 부모와 교사들을 위해, 전문 학술 논문 데이터를 학습한 대규모 언어 모델(LLM)을 개발하는 것을 목표로 합니다. <br>육아의 경우 부모의 경험이 없는 상황이 대부분이고, 빠른 대처가 필요하기에 육아 챗봇의 필요성을 찾을 수 있었습니다.

- **프로젝트 필요성**: 온라인에 산재한 부정확하거나 상업적인 육아 정보는 초보 부모에게 혼란을 야기할 수 있습니다.<br> 검증된 학술 자료를 기반으로 학습된 AI는 사용자가 과학적이고 신뢰도 높은 육아 지식에 쉽게 접근할 수 있도록 돕고, 자녀의 건강한 발달을 지원하는 데 중요한 역할을 할 수 있습니다.

- **주요 목표**: 육아·보육 정보 제공
육아 챗봇 시스템은 사용자에게 발달 단계, 건강 관리, 안전 지침 등 신뢰성 있는 정보를 제공합니다.<br>
사용자 맞춤형 응답
사용자의 질문 의도와 필요에 따라 정보를 제공하여 양육과 보육 경험을 최적화합니다.

<br>

## 3. 기술 스택 & 사용한 모델 

| 구분 | 기술 |
|---|---|
| 언어 | [![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/) |
| 개발 환경 | [![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/) [![Jupyter Notebook](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/) |
| 딥러닝/ML 라이브러리 | [![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/) [![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/) [![bitsandbytes](https://img.shields.io/badge/bitsandbytes-009485?style=for-the-badge)](https://github.com/TimDettmers/bitsandbytes) [![FAISS](https://img.shields.io/badge/FAISS-005CAB?style=for-the-badge)](https://faiss.ai/) |
| 데이터 처리/전처리 | [![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge)](https://www.langchain.com/) [![Regex](https://img.shields.io/badge/Regex-000000?style=for-the-badge)]() |
| 외부 API | [![Kakao API](https://img.shields.io/badge/Kakao%20API-FFCD00?style=for-the-badge&logo=kakao&logoColor=black)](https://developers.kakao.com/) [![HIRA API](https://img.shields.io/badge/HIRA%20API-005BAC?style=for-the-badge)](https://www.hira.or.kr/) |
| 협업/버전관리 | [![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/) |


<br>

#### | 사용한 모델 |

임베딩 모델 (Embedding Model)

선택 모델: BAAI/bge-m3
다국어 지원: 한국어·영어 모두 안정적인 임베딩 성능 제공
멀티 벡터 구조: Dense, Sparse, ColBERT 벡터 동시 지원 → 검색 품질 강화
정규화 지원: L2 Normalization으로 FAISS에서 Inner Product 검색 최적화
효율성: Colab T4 환경에서 대량 문서 임베딩 가능

선정 이유
1.	한국어 성능이 우수하고 다국어 확장성 보유
2.	오픈소스 + 상업적 사용 가능 라이선스 → 배포 제약 없음
3.	e5, KoSimCSE 등과 비교 실험 결과, RAG 환경에서 더 높은 Recall 성능 확인

<br>

대규모 언어 모델 (LLM)

Base 모델: Orion-zhen/Qwen2.5-7B-Instruct-Uncensored
	•	지시 따름(Instruct) 성능 강화된 Qwen2.5 계열 모델
	•	한국어/영어 지원: 국내 보육 문서에 적합하면서도 다국어 확장 가능
	•	Uncensored 버전: 컨텍스트 의존적 답변을 제약 없이 제공

양자화(Quantization) 적용
	•	방식: 4bit QLoRA 기반 양자화 (bitsandbytes 활용)
	•	이유: 메모리 사용량을 절반 이상 절감
			단일 GPU 환경에서도 추론 가능
			성능 저하를 최소화하면서 응답 지연을 줄임

배포 모델: WOOJINIYA/parentcare-bot-qwen2.5-7b
	•	Hugging Face Hub에 배포한 RAG 챗봇 리소스
	•	Base 모델(Orion-zhen/Qwen2.5-7B-Instruct-Uncensored)에 양자화 적용 + RAG 파이프라인 연동
	•	검색된 컨텍스트를 입력으로 받아 답변과 함께 출처를 반환
<br>
선정 이유
Hugging Face에서 제공하는 Leaderboard 지표들을 참고하여 모델을 비교 평가했습니다.<br>

	•	Average: 모델의 전반적 성능 평균<br>
	•	IFEval: 사람이 정한 규칙(시스템 프롬프트, 금지어, 출력 형식 등)을 얼마나 잘 따르는가<br>
	•	BBH (Big-Bench Hard): 복잡한 논리·추론 문제 해결 능력<br>
	•	MATH: 고등학교·대학교 수준의 수학 문제 해결 능력<br>
	•	GPQA: 대학원급 난이도의 구글 서치만으로는 해결하기 힘든 문제 해결 능력<br>
	•	MUSR: 다단계 추론 능력 (시간표·환승 계산 등)<br>
	•	MMLU: 수학, 과학 등 57개 과목에 대한 광범위한 지식 수준<br>

저희가 개발하려는 육아 특화 챗봇은 정확한 사실 전달과 더불어 안전 고지, 말투(존댓말/친절함), 출력 형식 유지 같은 규칙성이 무엇보다 중요합니다.
따라서 IFEval 점수가 높은 모델을 우선적으로 고려했습니다.
<br>
후보 모델 및 비교
	1.	Orion-zhen/Qwen2.5-7B-Instruct-Uncensored
	•	높은 IFEval
	•	한국어 학습 지원
	•	전반적으로 Average 점수도 높음

<br>

Vector Database

선택 DB: FAISS
	•	IndexFlatIP 기반으로 코사인 유사도 검색 수행
	•	Google Drive 연동을 통해 Colab 환경에서도 인덱스 파일(faiss.index, faiss.meta.json) 보관 가능
	•	장점:
	•	무료/오픈소스 → 비용 부담 없음
	•	GPU 가속 지원 → 대규모 데이터 확장성 확보
	•	Hugging Face, LangChain 등과 호환성 뛰어남

<br>

## 4. 시스템 아키텍처
<img width="930" height="522" alt="image" src="https://github.com/user-attachments/assets/5174be97-4225-40db-b9b6-2343f477de7f" />



<br>

## 5\. WBS
<img width="1319" height="665" alt="image" src="https://github.com/user-attachments/assets/c322baca-e38f-40ce-b0d8-a4234f4a4f97" />


<br>

## 6. 요구사항 명세서
<img width="1372" height="428" alt="image" src="https://github.com/user-attachments/assets/f2ada38c-6ccf-45af-a868-3c97f4115fb3" />





<br>

## 7. 수집한 데이터 및 전처리 요약

 데이터 소스

	1.	학술 논문 (PDF)
	•	국내 학회/저널, 학위논문 등
	•	영유아 발달, 부모-자녀 상호작용, 건강/위생 관련 연구

	2.	공공기관 가이드 (PDF)
	•	중앙육아종합지원센터, 아리누리 보육 가이드라인
	•	예방접종, 아동 안전, 보육 정책 관련 문서

	3.	커뮤니티 데이터 (TXT)
	•	부모 Q&A, 경험 공유 게시글 - 맘카페
	•	실제 육아 고민 사례 기반 데이터 - 아이사랑

<br>

 데이터 전처리 파이프라인


- 로컬
	1.	PDF 파일 준비
	•	./pdf_files/ 디렉토리에 원본 PDF 파일들을 저장
	•	코드 상단의 pdf_metadata에 각 파일명과 매핑될 메타데이터(title, id, category) 등록

	3.	텍스트 추출 (PyPDFLoader)
	•	각 PDF 페이지별 텍스트 추출 후 전체 문서를 하나의 문자열로 결합

	4.	정제 (clean_text)
	•	점/공백 반복 패턴 제거
	•	제어문자 제거
	•	목차 제거
	•	반복되는 머리말(예: Korea Institute of Child Care Education) 삭제
	•	한글/영문/숫자/일반 구두점 외 문자는 제거
	•	연속 공백 축소

	5.	청킹 (RecursiveCharacterTextSplitter)
	•	chunk_size=1000, chunk_overlap=200
	•	문단(\n\n) → 줄바꿈(\n) → 공백 → 문자 단위 순으로 분할
	•	긴 문서는 잘게 쪼개고, 짧은 문단은 합쳐서 적절한 크기 유지

	6.	JSONL 생성
	•	각 청크 단위를 { "text": <청크 텍스트>, "metadata": {title, id, category} } 구조로 저장
	•	최종 결과: output.jsonl (라인 단위 JSON)

	7.	출력 및 로그
	•	처리되지 않은 PDF(메타데이터 없음)는 경고 후 건너뜀
	•	전처리 완료 후 전체 청크 개수를 출력

- API

3) 텍스트 추출
	•	PyPDFLoader로 페이지 텍스트 추출

4) 클리닝(라이트)
	•	줄바꿈 보존을 전제로 최소화된 정규화:
	•	스페이스/탭 압축: [ \t]+ → ' '
	•	과도한 줄바꿈 축소: \n{3,} → \n\n
	•	목적: 문단/줄 경계를 최대한 살려 의미 기반 청킹에 피처로 쓰기

5) 의미 기반 청킹 (SemanticChunker)
	•	초기 분할 기준: ["\n\n", "\n", "。", "！", "？", ".", "!", "?", " "]<br>
	•	SemanticChunker(LangChain experimental) + OpenAI 임베딩으로 breakpoint 계산<br>
	•	breakpoint_threshold_type="percentile", breakpoint_threshold_amount=95 (예시)<br>
	•	유사도가 급격히 떨어지는 지점(=주제 전환)에 절취선을 만든다.<br>
	•	길이 제어: 토큰 기준<br>
	•	chunk_size_tokens=400 ~ 600, chunk_overlap_tokens=80~120<br>
	•	Fallback: 의미 경계가 애매할 경우 RecursiveCharacterTextSplitter로 토큰 기준 분할


<br>

## 8. DB 연동 구현 코드 (링크만)
https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN17-3rd-4Team/blob/main/Model_DB.ipynb

<br>

## 9. 테스트 계획 및 결과 보고서
<img width="446" height="229" alt="스크린샷 2025-09-25 오후 2 14 16" src="https://github.com/user-attachments/assets/e8e7c086-c806-4a06-bd4b-ee6b0aecbe49" />
<br>
<img width="802" height="150" alt="스크린샷 2025-09-25 오후 2 13 52" src="https://github.com/user-attachments/assets/f21d4b5e-be73-43f5-a152-5dc190a7f41a" />
<br>
<img width="508" height="674" alt="스크린샷 2025-09-25 오후 2 13 45" src="https://github.com/user-attachments/assets/50b274b8-d34e-4662-99d4-bcf0f938d46a" />



<br>

## 10. 진행 과정 중 프로그램 개선 노력

i. 초기 단계에서는 파인튜닝을 직접 시도했으나, 모델이 기대만큼 데이터를 반영하지 못했고 응답 속도도 크게 느려지는 문제가 발생함.

ii. 검토 결과, 파인튜닝은 모델 크기 대비 효과가 제한적이고 운영 비용(시간·리소스)도 크다고 판단하여, 모든 데이터를 RAG 기반 검색-생성 구조로 전환하기로 결정함.

iii. 데이터 출처가 논문, 연구보고서, 부모지원자료, 커뮤니티 등으로 다양하여, 처음에는 메타데이터 구조가 제각각이었음. 이로 인해 검색과 출처 인용 단계에서 충돌이 발생 → 공통 메타데이터 스키마(id, title, category, source 등) 를 정의하고 전처리 단계에서 강제 매핑하도록 개선함.

iv. RAG의 핵심은 “검색된 문서가 얼마나 자연스럽고 연속적인 문맥 단위로 회수되느냐”에 달려 있었음. 따라서 전처리 과정에서 줄바꿈 보존, 불필요한 제어문자 제거, 목차·머리말 삭제, 문단/문장 경계 기반 청킹 등 다양한 개선을 반복함.

v. 초기에는 단순 문자 단위 청킹으로 인해 문맥이 잘려나가는 경우가 많았으나, 이후 분리자 우선순위(문단 → 줄 → 문장부호 → 공백 → 문자)를 적용하여 회수된 청크가 보다 자연스러운 의미 단위가 되도록 개선함.

vi. 최종적으로는, 전처리와 청킹을 통해 데이터가 안정적으로 JSONL로 변환되고, FAISS 기반 벡터 검색에서 정확성과 일관성 있는 검색 품질을 달성할 수 있었음. 이로써 모델 응답 신뢰도가 높아지고, 파인튜닝 대비 훨씬 빠른 응답 속도를 유지할 수 있었음.


<br>

## 11. 수행결과(테스트/시연 페이지)
<img width="871" height="707" alt="2d06c62105efa525" src="https://github.com/user-attachments/assets/ed605c9b-85bd-4939-be91-58349c808175" />
<img width="821" height="433" alt="3dd8efd805be4daf" src="https://github.com/user-attachments/assets/63079584-7fcb-4050-bf16-2460f71a3710" />
<img width="788" height="498" alt="989c12d378be23c2" src="https://github.com/user-attachments/assets/1c9f2e3e-c1ef-4e47-9a75-a328b3f26760" />
<img width="787" height="528" alt="2881cb2f977cf532" src="https://github.com/user-attachments/assets/5cdc2ee6-1ed9-4eca-bf44-886da6ace38f" />
<img width="797" height="425" alt="30999f9e75ab7f5a" src="https://github.com/user-attachments/assets/8e2f0d06-7d38-4993-9b0f-445c3e7eef2f" />
<img width="747" height="302" alt="368143c3f7f36a66" src="https://github.com/user-attachments/assets/578d0559-ad2e-4d39-9402-7113d3daab09" />
<img width="757" height="562" alt="c6c7700b233b083b" src="https://github.com/user-attachments/assets/051cd778-710b-4a80-935b-e0ad8f816e1c" />
<img width="792" height="707" alt="e2cb78b4dafc9591" src="https://github.com/user-attachments/assets/c990e0dd-dd51-413d-b15a-fc02921e933b" />


<br>

## 12. 한 줄 회고

  - **김민균**: 크롤링을 하면서 크롤링 규정에 걸린 곳들이 많이 있어서 힘이 들었고 프로젝트를 하면서 한번더 부족하다는 것을 느끼고 좀 더 노력을 하여서 앞으로는 여러가지의 크롤링과 함께 더 많은 것들이 가능 하도록 노력하겠습니다.
  - **김세한**: 임베딩 모델과 LLM 모델 선정 단계부터 크롤링 및 전처리를 통한 데이터 구축, 이를 RAG와 파인튜닝용으로 가공하여 벡터DB를 만들고 FAISS를 활용해 RAG를 적용하는 전 과정까지 직접 수행하며 전체 흐름을 이해할 수 있었다. 다만 파인튜닝 코드까지 준비했음에도 불구하고 시간적 제약으로 학습을 완료하지 못한 점은 아쉬움으로 남는다. 향후에는 파인튜닝까지 마무리한 모델을 RAG와 결합하여 보다 완성도 높은 시스템을 구현하고자 한다.
  - **김수현**: 웹 크롤링을 통해 데이터를 수집하는 과정에서 스스로의 부족한 점을 돌아보게 되었습니다. 일부 데이터를 수집했는데 막상 분석하고 활용하기에는 어려운 데이터가 많아 아쉬움이 남았습니다. 이는 저의 경험 부족에서 비롯된 것이라 생각하며, 앞으로는 데이터의 품질과 활용성을 높일 수 있는 방법에 대해 더 깊이 공부하고 노력하는 자세를 갖겠습니다.
  - **정의중**: 데이터 수집과 전처리를 맡으며, 팀이 신뢰할 수 있는 기반을 만드는 일이 얼마나 중요한지 깨달았습니다. RAG의 성능을 높이기 위해 작은 세부 개선을 반복하는 과정에서, 팀원들의 아이디어와 노력이 더해져 큰 성과로 이어졌습니다.
  - **최우진**: 초기 기획부터 RAG 아키텍처를 핵심 전략으로 설정하고, 신뢰성 있는 답변을 위해 다양한 육아 관련 문서를 수집, 정제, 그리고 의미 단위로 청킹하여 벡터 DB를 구축해보았습니다. 이후, 검색과 생성에 최적화된 임베딩 모델과 LLM을 선정하여 정교한 프롬프트 엔지니어링으로 LLM의 답변 품질을 끌어올리는 성과를 얻었습니다. 최종적으로 데이터 파이프라인 설계부터 모델 적용까지 전 과정을 경험하며 AI 서비스 개발 역량을 키울 수 있었습니다.




# 👶 육아 도우미(ParentCare) 프로젝트

**도담이(챗봇)**를 중심으로 육아 정보를 신속·친절하게 안내하는 경량 웹 서비스입니다. 본 버전은 **챗봇 + 인증(로그인/회원가입) + 홈 랜딩**으로 구성됩니다.

---

## 목차

1. [팀 소개](#-팀-소개)
2. [프로젝트 개요](#-프로젝트-개요)
3. [작업 계획](#-작업-계획)
4. [기술 스택](#-기술-스택)
5. [시스템 구성도](#-시스템-구성도)
6. [핵심 기능](#-핵심-기능)
7. [요구사항 정의](#-요구사항-정의)
8. [API 설계](#-api-설계)
9. [화면 설계서](#-화면-설계서)
10. [테스트 계획 및 결과](#-테스트-계획-및-결과)
11. [실행 방법](#-실행-방법)
12. [폴더 구조](#-폴더-구조)
13. [브랜치/이슈/PR 운영](#-브랜치이슈pr-운영)
14. [향후 과제](#-향후-과제)

---

## 1. 팀 소개

| 이름  | GitHub ID  |
| --- | ---------- |
| 김민균 | <a href="https://github.com/alswhitetiger">@alswhitetiger</a></td> |
| 김세한 | <a href="https://github.com/kimsehan11">@kimsehan11</a></td> |
| 김수현 | <a href="https://github.com/K-SH98">@K-SH98</a></td> |
| 정의중 | <a href="https://github.com/uii42">@uii42</a></td> |
| 최우진 | <a href="https://github.com/CHUH00">@CHUH00</a></td> |

---

## 2. 프로젝트 개요

### 2-1) 문제정의

* 육아 정보가 분산되어 있어 **즉시 접근**이 어렵고, 긴급 상황에서 **요약된 가이드**가 필요함.

### 2-2) 목표

* 챗봇 **도담이**를 통해 질문-답변을 **빠르게 제공**
* 신뢰 가능한 출처 기반 응답(출처/갱신일 표시)
* 로그인/회원가입으로 **개인화 기본 토대** 제공 (향후 알림/히스토리 확장 대비)

### 2-3) 범위(Scope)

* 포함: 홈 랜딩, 로그인, 회원가입, **챗봇(대화/저장)**

---

## 3. 작업 계획

> **기간: 2일(데모/발표용 스프린트)**

| Day    | 마일스톤           | 상세 작업                                         | 산출물                  |
| ------ | -------------- | --------------------------------------------- | -------------------- |
| 1일차 오전 | 리포 세팅          | README/이슈·PR 템플릿/브랜치 규칙 적용, CI 스켈레톤           | 초기 커밋, CI 통과         |
| 1일차 오후 | 화면 마크업 1       | **메인 홈(scr-main-01)**, **로그인(scr-signin-01)** | 화면 초안, 라우팅           |
| 2일차 오전 | 화면 마크업 2       | **회원가입(scr-signup-01)**, **챗봇(scr-chat-01)**  | 화면 초안, 상태/폼 검증       |
| 2일차 오후 | API 목/연동 & 테스트 | 가짜 API/JSON 서버, 가입/로그인 흐름, 간단 E2E             | 데모 빌드(v0.1), 테스트 결과표 |

* 위험/대응: 데이터 지연 ▶ 목API로 대체, 이미지 미확정 ▶ 플레이스홀더 사용\

---

## 4. 기술 스택

**개발 도구**
[![VS Code](https://img.shields.io/badge/VS%20Code-007ACC?logo=visualstudiocode\&logoColor=white)](#)
[![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker\&logoColor=white)](#)
[![Docker Hub](https://img.shields.io/badge/Docker%20Hub-2496ED?logo=docker\&logoColor=white)](#)
[![Git](https://img.shields.io/badge/Git-F05032?logo=git\&logoColor=white)](#)
[![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-2088FF?logo=githubactions\&logoColor=white)](#)
[![Postman](https://img.shields.io/badge/Postman-FF6C37?logo=postman\&logoColor=white)](#)

**개발 언어**
[![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?logo=typescript\&logoColor=white)](#)
[![Python](https://img.shields.io/badge/Python-3776AB?logo=python\&logoColor=white)](#)

**벡터 DB**
[![FAISS](https://img.shields.io/badge/FAISS-0055FF?logo=vectorworks\&logoColor=white)](#)

**사용하는 모델**
[![GPT-4o mini](https://img.shields.io/badge/GPT--4o%20mini-412991?logo=openai\&logoColor=white)](#)
[![text-embedding-3-large](https://img.shields.io/badge/text--embedding--3--large-412991?logo=openai\&logoColor=white)](#)

**서버**
[![Django](https://img.shields.io/badge/Django-092E20?logo=django\&logoColor=white)](#)
[![Uvicorn](https://img.shields.io/badge/Uvicorn-1F9AFE?logo=fastapi\&logoColor=white)](#)
[![Gunicorn](https://img.shields.io/badge/Gunicorn-499848?logo=gunicorn\&logoColor=white)](#)
[![Nginx](https://img.shields.io/badge/Nginx-009639?logo=nginx\&logoColor=white)](#)

**서비스 제공자**
[![AWS](https://img.shields.io/badge/AWS-FF9900?logo=amazonwebservices\&logoColor=white)](#)

**데이터베이스**
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?logo=postgresql\&logoColor=white)](#)

**협력 도구**
[![Slack](https://img.shields.io/badge/Slack-4A154B?logo=slack\&logoColor=white)](#)
[![Notion](https://img.shields.io/badge/Notion-000000?logo=notion\&logoColor=white)](#)
[![Figma](https://img.shields.io/badge/Figma-F24E1E?logo=figma\&logoColor=white)](#)
[![Google Drive](https://img.shields.io/badge/Google%20Drive-4285F4?logo=googledrive\&logoColor=white)](#)


---

## 5. 시스템 구성도

```
[Client: Web/Mobile]
        |
        v
[API Gateway / Nginx]
        |
        v
[Backend: Django/DRF] --- [Auth(JWT/OAuth)]
        |                       |
        v                       v
   [PostgreSQL]            [Object Storage(S3)]
        |
        v
[ETL/Data Jobs] <--- [공공데이터/의료기관/보건소 API]
```

아키텍처 다이어그램(이미지):\

---

## 6. 핵심 기능

* **도담이 챗봇**: 자연어 질의에 대한 요약형 답변, 출처/갱신일 표기, 최근 대화 **.txt 저장**
* **인증**: 이메일 로그인/회원가입(닉네임/이메일/비밀번호 규칙/인증번호)
* **홈 랜딩**: 서비스 소개, 챗봇 시작 CTA, 푸터(운영사/저작권)

---

## 7. 요구사항 정의

> 우선순위: **Must/Should/Could**  · 수용 기준(AC)은 검증 가능한 문장으로 기술

### 7-1) 기능 요구사항(요약표)

| ID           | 영역   | 요구사항 명   | 핵심 내용                                            | 우선순위   | AC(수용 기준)                              |
| ------------ | ---- | -------- | ------------------------------------------------ | ------ | -------------------------------------- |
| REQ-ACC-001  | 가입   | 이메일 회원가입 | 닉네임(2~20), 이메일 형식, 인증메일 6자리, 비밀번호 규칙(영+숫+특 10자↑) | Must   | 유효 데이터 입력 시 1분 내 인증메일 수신·검증 통과         |
| REQ-ACC-002  | 로그인  | 이메일 로그인  | 이메일/비밀번호 검증, 오류 메시지, 성공 시 홈 리다이렉트                | Must   | 잘못된 조합 시 에러 토스트, 성공 시 세션 생성 및 홈 이동     |
| REQ-ACC-003  | 로그아웃 | 세션 종료    | 토큰/스토리지 삭제 후 랜딩 이동                               | Must   | 로그아웃 직후 보호 페이지 접근 차단                   |
| REQ-CHAT-001 | 대화   | 챗봇 질의/응답 | 사용자 입력 → 답변 렌더, 입력중 애니메이션                        | Must   | 10회 연속 질의에도 UI 렉 없이 표시                 |
| REQ-CHAT-002 | 저장   | 대화 저장    | 최근 대화를 **.txt**로 저장                              | Should | 파일명 규칙 `dodam_YYYYMMDD_HHMM.txt`로 다운로드 |
| REQ-CHAT-003 | 신뢰성  | 출처/갱신일   | 답변 하단에 출처/업데이트일 표기                               | Should | 95% 이상 응답에 메타 표기 노출                    |

---

## 8. API 설계

> 핵심 엔드포인트만 명시 (예시)

| 메서드  | 경로                            | 설명                       |
| ---- | ----------------------------- | ------------------------ |
| POST | `/api/auth/signup`            | 회원가입(닉네임/이메일/비밀번호)       |
| POST | `/api/auth/login`             | 로그인(JWT 발급)              |
| POST | `/api/chat/ask`               | 질문 전송 → 답변 반환(출처/갱신일 포함) |
| GET  | `/api/chat/export?format=txt` | 최근 대화 내보내기               |

응답 예시(챗봇):

```json
{
  "answer": "모유 수유 중 음식 권장사항은...",
  "sources": [
    {"title": "보건복지부 가이드", "url": "https://...", "updated_at": "2025-10-20"}
  ]
}
```

---- | ----------------------------- | ------------------------ |
| POST | `/api/auth/signup`            | 회원가입(닉네임/이메일/비밀번호)       |
| POST | `/api/auth/login`             | 로그인(JWT 발급)              |
| POST | `/api/chat/ask`               | 질문 전송 → 답변 반환(출처/갱신일 포함) |
| GET  | `/api/chat/export?format=txt` | 최근 대화 내보내기               |

응답 예시(챗봇):

```json
{
  "answer": "모유 수유 중 음식 권장사항은...",
  "sources": [
    {"title": "보건복지부 가이드", "url": "https://...", "updated_at": "2025-10-20"}
  ]
}
```


응답 예시:

```json
{
  "items": [
    {
      "id": 123,
      "name": "행복소아과의원",
      "lat": 37.51,
      "lng": 127.02,
      "night_service": true,
      "phone": "02-123-4567"
    }
  ],
  "total": 1
}
```

---

## 9. 화면 설계서

첨부된 시안 기반으로 **요소 번호-설명 매핑**을 표로 정리했습니다.

### 9-1) 메인 홈 화면 (`scr-main-01`)

| No | 설명                                   |
| -- | ------------------------------------ |
| 1  | 상단 로고/서비스명(베이비가이드) – 클릭 시 홈          |
| 2  | 상단 링크: 도담이(바로가기), 회원가입, 로그인          |
| 3  | 히어로 타이틀/설명 문구                        |
| 4  | **CTA**: “도담이, 육아 도우미 챗봇” → 챗봇 화면 이동 |
| 5  | 푸터: 운영사/주소/저작권 표시                    |

---

### 9-2) 로그인 화면 (`scr-signin-01`)

| No | 설명                                    |
| -- | ------------------------------------- |
| 1  | 이메일 입력(형식 검증)                         |
| 2  | 비밀번호 입력(보기 토글)                        |
| 3  | 로그인 버튼 – 성공 시 홈 이동, 실패 시 오류 토스트(3-1)  |
| 4  | 회원가입/비밀번호 찾기 링크                       |
| 5  | 오류 모달(3-1): “이메일 또는 비밀번호가 일치하지 않습니다.” |

---

### 9-3) 회원가입 화면 (`scr-signup-01`)

| No | 설명                              |
| -- | ------------------------------- |
| 1  | 닉네임 입력(2~20자) + 중복확인(2-1/2-2)   |
| 3  | 이메일 입력 + 형식 검증                  |
| 4  | 인증요청 버튼 → 메일 6자리 발송(4-1/4-2 결과) |
| 5  | 인증번호 입력 및 검증                    |
| 7  | 비밀번호 규칙 안내(영+숫+특 10자↑)          |
| 8  | 비밀번호 확인 일치 검사                   |
| 9  | 회원가입 완료(9-1)                    |
| 10 | 로그인 화면 이동 링크                    |
| 11 | 결과 모달들(중복/인증/완료)                |

---

### 9-4) 챗봇 화면 (`scr-chat-01`)

| No | 설명                            |
| -- | ----------------------------- |
| 1  | 도담이 아바타/이름 표시                 |
| 2  | 사용자 버블, 우측 정렬                 |
| 3  | 답변 버블, 장문 스크롤                 |
| 4  | 입력중 애니메이션(typing)             |
| 5  | 입력창 플레이스홀더 "도담이에게 물어보세요 :)"   |
| 6  | 전송 버튼                         |
| 7  | 최근 대화 저장(.txt), 저장 성공 알림(7-1) |
| 8  | 저장 완료 모달                      |

---|---|
| 1 | 서비스 로고/문구 : 클릭 시 홈 이동 |
| 2 | 도입부 배지/탭 |
| 3 | **시작하기** 버튼 : 회원가입/온보딩으로 이동 |
| 4 | 로그인 버튼 : 로그인 페이지로 이동 |
| 5 | 메인 히어로 CTA : 챗봇 진입 혹은 핵심 기능 랜딩 |
| 6 | 푸터 : 운영사/저작권/서비스 제공자 정보 노출 |

---

### 9-2) 로그인 화면 (`scr-signin-01`)

| No | 설명                                                               |
| -- | ---------------------------------------------------------------- |
| 1  | 이메일 입력 : 형식 검증, 포맷 힌트([user@domain.com](mailto:user@domain.com)) |
| 2  | 비밀번호 입력 : 마스킹/보기 토글                                              |
| 3  | 로그인 버튼 : 성공 시 메인, 실패 시 오류 안내(3-1)                                |
| 4  | 회원가입/비밀번호 찾기 링크                                                  |
| 5  | 오류 모달(3-1) : 자막형 알림 + 확인 버튼                                      |

---

### 9-3) 회원가입 화면 (`scr-signup-01`)

| No | 설명                                  |
| -- | ----------------------------------- |
| 1  | 닉네임 입력(2~20자) + 중복확인(2-1/2-2 결과 알림) |
| 3  | 이메일 입력 + 형식 검증                      |
| 4  | 인증요청 버튼 : 6자리 메일 발송(4-1/4-2 결과)     |
| 5  | 인증번호 입력(6자리) + 검증                   |
| 7  | 비밀번호(10자↑, 영+숫+특) 입력 및 규칙 표시        |
| 8  | 비밀번호 확인 일치 검사                       |
| 9  | 회원가입 완료 버튼 : 모든 검증 통과 시 완료(9-1)     |
| 10 | 기존 계정 이동 링크(로그인)                    |
| 11 | 결과 알림 모달들 : 중복/인증/완료 등              |

---

### 9-4) 챗봇 화면 (`scr-chat-01`)

| No | 설명                             |
| -- | ------------------------------ |
| 1  | 봇 식별(아바타/이름)                   |
| 2  | 사용자 말풍선, 정렬 기준 일관              |
| 3  | 답변 영역(가변 높이, 장문 스크롤)           |
| 4  | 입력 중 표시 애니메이션(typing)          |
| 5  | 입력창 플레이스홀더 "도담이에게 물어보세요 :)"    |
| 6  | 보내기 버튼                         |
| 7  | 최근 대화 저장(.txt) 버튼 + 완료 알림(7-1) |
| 8  | 저장 완료 모달(확인)                   |

---

## 10. 테스트 계획 및 결과

### 10-1) 테스트 계획

| ID           | 시나리오        | 절차                    | 기대 결과                          |
| ------------ | ----------- | --------------------- | ------------------------------ |
| TC-LOGIN-01  | 이메일 로그인 성공  | 올바른 계정 입력 → 로그인       | 홈 이동, 세션 생성                    |
| TC-LOGIN-02  | 이메일 로그인 실패  | 잘못된 비밀번호 입력           | 오류 토스트(3-1) 노출                 |
| TC-SIGNUP-01 | 인증 메일 발송/검증 | 이메일 입력 → 인증요청 → 코드 입력 | 1분 내 메일 수신·검증 통과               |
| TC-CHAT-01   | 기본 질의/응답    | 질문 입력 → 응답 확인         | 3s p95 이내 답변 표시                |
| TC-CHAT-02   | 대화 저장       | 저장 버튼 클릭              | `dodam_YYYYMMDD_HHMM.txt` 다운로드 |

### 10-2) 결과(예시)

| ID          | 결과   | 비고       |
| ----------- | ---- | -------- |
| TC-LOGIN-01 | Pass | 200ms 응답 |
| TC-CHAT-01  | Pass | 2.5s p95 |

---|---|---|---|
| TC-001 | 위치 기반 병원 조회 | 위치 허용 → 야간 필터 On | 야간 진료 병원만 표시 |
| TC-010 | 접종 알림 | DOB 입력 → 접종 달 도래 | 알림 수신(앱/메일) |
| TC-020 | 신고/블라인드 | 동일 게시물 3회 신고 | 자동 블라인드 처리 |

### 10-2) 결과(예시)

| ID     | 결과   | 비고            |
| ------ | ---- | ------------- |
| TC-001 | Pass | 응답 450ms p95  |
| TC-010 | Pass | 메일 도착 30초 내   |
| TC-020 | Pass | 운영자 페이지 로그 확인 |


---

## 11. 실행 방법

### 11-1) 환경 변수

`.env` (예시)

```
DATABASE_URL=postgresql://user:pass@host:5432/parentcare
JWT_SECRET=replace_me
MAPS_API_KEY=replace_me
```

### 11-2) 로컬(예: Node + Django 혼합 리포일 때)

```bash
# Frontend
cd frontend
npm i
npm run dev

# Backend
cd backend
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

### 11-3) 도커(옵션)

```bash
docker compose up -d --build
```

---

## 12. 폴더 구조

```
.
├─ frontend/               # React, Vite
│  ├─ src/
│  └─ ...
├─ backend/                # Django/DRF
│  ├─ app/
│  └─ ...
├─ docs/
│  ├─ architecture.png
│  └─ ui/
│     ├─ home.png
│     ├─ signin.png
│     ├─ signup.png
│     └─ chat.png
├─ .github/
│  ├─ ISSUE_TEMPLATE/
│  │  ├─ bug_report.yml
│  │  └─ feature_request.yml
│  └─ pull_request_template.md
├─ .env.example
└─ README.md
```

---

## 14. 향후 과제

* 푸시 알림(모바일/PWA) 정교화, 다자녀 프로필 지원
* 의료기관/약국 실시간 혼잡도·대기시간 연동
* 신뢰성 검증(출처) 자동 업데이트 및 변경 알림
* 챗봇 증강(RAG) 정답률 지표화, 금칙어/응급 가이드 보강
* 접근성 개선(색대비/키보드 내비/스크린리더)

---

## 한줄 회고록
* 김민균:
* 김세한:
* 김수현:
* 정의중:
* 최우진:
