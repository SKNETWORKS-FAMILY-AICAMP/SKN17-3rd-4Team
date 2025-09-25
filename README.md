# 육아 논문 데이터 기반 LLM RAG 프로젝트

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

