# 육아 LLM 프로젝트

본 프로젝트는 육아·보육 관련 내외부 문서를 기반으로 한 질의응답 시스템을 구축하는 것을 목표로 합니다.
LLM에 직접 파인튜닝을 적용하지 않고, Retrieval-Augmented Generation(RAG) 방식을 통해 문서 검색 결과를 컨텍스트로 활용하여 신뢰성 있는 답변을 제공합니다.

<br>

## 1 👨‍👩‍👧‍👦 팀 소개

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

- **프로젝트 소개**: 본 프로젝트는 넘쳐나는 육아 정보 속에서 신뢰할 수 있는 정보를 찾기 어려운 부모와 교사들을 위해, 전문 학술 논문 데이터를 학습한 대규모 언어 모델(LLM)을 개발하는 것을 목표로 합니다.<br>육아에 경우 부모의 경험이 없는 상황이 많으며 빠른 대처가 필요하기에 육아 챗봇의 필요성을 찾아볼 수 있습니다.

- **프로젝트 필요성**: 온라인에 산재한 부정확하거나 상업적인 육아 정보는 초보 부모에게 혼란을 야기할 수 있습니다.<br>검증된 학술 자료를 기반으로 학습된 AI는 사용자가 과학적이고 신뢰도 높은 육아 지식에 쉽게 접근할 수 있도록 돕고, 자녀의 건강한 발달을 지원하는 데 중요한 역할을 할 수 있습니다.

- **주요 목표**: 육아·보육 정보 제공
육아 챗봇 시스템은 사용자에게 발달 단계, 건강 관리, 안전 지침 등 신뢰성 있는 정보를 제공합니다.<br>
사용자 맞춤형 응답
사용자의 질문 의도와 필요에 따라 정보를 제공하여 양육과 보육 경험을 최적화합니다.

<br>

## 3. 기술 스택 & 사용한 모델

| 분류 | 기술/도구 |
|---|---|
| 언어 | [![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/) |
| 개발 환경 | [![Jupyter Notebook](https://img.shields.io/badge/Jupyter%20Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/) |
| 딥러닝/ML 라이브러리 | [![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/) [![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/) [![bitsandbytes](https://img.shields.io/badge/bitsandbytes-009485?style=for-the-badge)](https://github.com/TimDettmers/bitsandbytes) |
| 협업 툴 | [![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/) |

<br>

#### 임베딩 모델 (Embedding Model)

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

#### 대규모 언어 모델 (LLM)

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

#### Vector Database

선택 DB: FAISS
	•	IndexFlatIP 기반으로 코사인 유사도 검색 수행
	•	Google Drive 연동을 통해 Colab 환경에서도 인덱스 파일(faiss.index, faiss.meta.json) 보관 가능
	•	장점:
	•	무료/오픈소스 → 비용 부담 없음
	•	GPU 가속 지원 → 대규모 데이터 확장성 확보
	•	Hugging Face, LangChain 등과 호환성 뛰어남

<br>


## 4. 시스템 아키텍처

<img width="930" height="522" alt="image" src="https://github.com/user-attachments/assets/e8d37425-4c98-4f16-a934-f8daec197740" />


<br>

## 5\. WBS
<img width="1319" height="665" alt="image" src="https://github.com/user-attachments/assets/c322baca-e38f-40ce-b0d8-a4234f4a4f97" />


<br>

## 6\. 요구사항 명세서





<br>

## 7\. 수집한 데이터 및 전처리 요약

#### 데이터 소스

1.	학술 논문 (PDF/HWPX)
	•	국내 학회/저널, 학위논문 등
	•	영유아 발달, 부모-자녀 상호작용, 건강/위생 관련 연구
	2.	정부·공공기관 가이드 (PDF/HTML)
	•	보건복지부, 질병관리청, 지자체 제공 보육 가이드라인
	•	예방접종, 아동 안전, 보육 정책 관련 문서
	3.	커뮤니티 데이터 (HTML/TXT)
	•	부모 Q&A, 경험 공유 게시글
	•	실제 육아 고민 사례 기반 데이터 (PII 제거 후 활용)

<br>

 #### 데이터 전처리 파이프라인

1) 텍스트 추출
	•	PDF: pdfminer.six 활용 → 본문/각주/참고문헌 분리 태깅
	•	HWPX: xml.etree.ElementTree 기반 파서 → 스타일 태그 제거 후 텍스트만 추	출
	•	HTML: BeautifulSoup 사용 → 본문 영역만 선택, 광고/네비게이션 제거

2) 정규화 및 클리닝
	•	불필요 개행, 연속 공백, 특수문자 제거
	•	페이지 번호, 머리말/꼬리말, 중복 구문 제거
	•	표/그림 캡션은 본문과 함께 유지 → 근거 인용 시 활용 가능

3) 문서 메타데이터 생성
	•	문서별 고유 ID 부여 (doc_YYYYMMDD_xxxxx)
	•	기본 필드: id, title, doc_type(paper/guideline/community), lang, license
	•	출처 필드: url, doi, publisher

4) 청킹(Chunking)
	•	문단/문장 단위 분할 + 토큰 길이 제한 (기본 500 tokens, overlap 100 tokens)
	•	긴 문단은 추가 분할, 짧은 문단은 병합 처리
	•	chunk_id, chunk_index, page_from/to, section 필드 생성

5) 품질 필터링
	•	커뮤니티 데이터 내 이름, 전화번호, 이메일, 주소 등 식별자 → 정규식 기반 마스킹
	•	문장 수 3개 미만, 의미 없는 짧은 글은 제거
	•	유사도 ≥ 0.96인 중복 텍스트는 deduplication 처리


<br>

## 8. DB 연동 구현 코드 (링크만)



<br>

## 9. 테스트 계획 및 결과 보고서



<br>

## 10. 진행 과정 중 프로그램 개선 노력




<br>


## 11. 수행결과(테스트/시연 페이지)


<br>



## 12\. 한줄 회고록

  - **김민균**: QLoRA와 `unsloth`를 활용해 제한된 GPU 환경에서도 대규모 모델을 효율적으로 파인튜닝할 수 있다는 점이 인상 깊었습니다.
  - **김세한**: 신뢰도 높은 데이터셋을 구축하기 위해 논문을 정제하는 과정이 까다로웠지만, 모델 성능의 기반이 된다는 점에서 큰 보람을 느꼈습니다.
  - **김수현**: 웹 크롤링을 통해 데이터를 수집하는 과정에서 스스로의 부족한 점을 돌아보게 되었습니다. 일부 데이터를 수집했는데 막상 분석하고 활용하기에는 어려운 데이터가 많아 아쉬움이 남았습니다. 이는 저의 경험 부족에서 비롯된 것이라 생각하며, 앞으로는 데이터의 품질과 활용성을 높일 수 있는 방법에 대해 더 깊이 공부하고 노력하는 자세를 갖겠습니다.
  - **정의중**: 명확한 목표를 공유하고 각자의 역할에 최선을 다한 덕분에 시너지를 낼 수 있었습니다. 최고의 팀워크였습니다\!
  - **최우진**: LLM 파인튜닝의 전 과정을 직접 경험하며 이론으로만 알던 개념들을 체득할 수 있었던 값진 프로젝트였습니다.

