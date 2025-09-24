## 👨‍👩‍👧‍👦 팀 소개

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


# 👶 육아 논문 데이터 기반 LLM Fine-tuning 프로젝트

**`WOOJINIYA/parentcare-bot-qwen2.5-7b`** 모델을 활용하여 육아 관련 전문 지식을 학습시키는 파인튜닝(Fine-tuning) 프로젝트입니다. 국내 학술 논문 데이터를 통해 모델이 육아 분야의 질문에 더 정확하고 전문적인 답변을 생성하도록 하는 것을 목표로 합니다.

<br>

## 1. 프로젝트 개요

- **프로젝트 소개**: 본 프로젝트는 넘쳐나는 육아 정보 속에서 신뢰할 수 있는 정보를 찾기 어려운 부모와 교사들을 위해, 전문 학술 논문 데이터를 학습한 대규모 언어 모델(LLM)을 개발하는 것을 목표로 합니다. `WOOJINIYA/parentcare-bot-qwen2.5-7b` 모델은 육아에 대한 깊이 있는 질문에 학술적 근거를 바탕으로 전문적인 답변을 제공하는 인공지능 챗봇의 기반이 됩니다.
- **프로젝트 필요성**: 온라인에 산재한 부정확하거나 상업적인 육아 정보는 초보 부모에게 혼란을 야기할 수 있습니다. 검증된 학술 자료를 기반으로 학습된 AI는 사용자가 과학적이고 신뢰도 높은 육아 지식에 쉽게 접근할 수 있도록 돕고, 자녀의 건강한 발달을 지원하는 데 중요한 역할을 할 수 있습니다.
- **주요 목표**: 육아(유아 발달, 상호작용, 사회성 등) 관련 텍스트 데이터셋을 구축하고, 이를 기반으로 대규모 언어 모델(LLM)을 미세조정하여 특정 도메인에 특화된 모델 개발
- **활용 모델**: `WOOJINIYA/parentcare-bot-qwen2.5-7b`
- **핵심 기술**: `Transformers`, `PEFT`, `bitsandbytes` (QLoRA)
- **데이터셋**: 국내 육아 관련 학술 논문에서 추출한 텍스트로 구성된 `output.jsonl` 파일

<br>

## 2. 개발 환경 및 라이브러리

| 분류 | 기술/도구 |
|---|---|
| 언어 | [![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/) |
| 개발 환경 | [![Jupyter Notebook](https://img.shields.io/badge/Jupyter%20Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/) |
| 딥러닝/ML 라이브러리 | [![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/) [![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/) [![bitsandbytes](https://img.shields.io/badge/bitsandbytes-009485?style=for-the-badge)](https://github.com/TimDettmers/bitsandbytes) |
| 협업 툴 | [![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/) |

<br>

## 3. 기술 스택 및 구현 내용

#### 가. 모델 및 토크나이저 로드
- **모델**: 허깅페이스(Hugging Face)에 공개된 `unsloth/qwen2.5-7b-instruct-bnb-4bit` 모델을 기반으로 파인튜닝을 진행했습니다.
- **모델 선정 기준**
  - **우수한 한국어 성능**: Qwen2 모델은 다양한 최신 LLM 벤치마크에서 높은 성능을 보이며, 특히 한국어 이해 및 생성 능력에서 강점을 가집니다.
  - **효율적인 파인튜닝**: `unsloth` 라이브러리는 Qwen2 모델에 최적화되어 있어, 기존보다 약 2배 빠른 학습 속도와 60% 절감된 메모리 사용량을 제공하여 효율적인 미세조정이 가능합니다.
  - **명령어 처리 능력**: Instruct(명령어) 기반으로 사전 훈련되어, 논문 요약과 같은 특정 과업 지시에 대한 이해도가 높아 파인튜닝에 적합합니다.
  - **합리적인 모델 크기**: 7B 파라미터 모델은 제한된 리소스 환경에서도 효과적인 파인튜닝과 추론이 가능하여, 개인 및 소규모 팀 프로젝트에 적합합니다.
- **양자화 (Quantization)**: 메모리 부족 문제(`RuntimeError`)를 해결하고 효율적인 학습을 위해 `BitsAndBytesConfig`를 사용하여 4-bit QLoRA 양자화를 적용했습니다.

#### 나. 데이터 전처리
- **데이터 로드**: `output.jsonl` 파일을 로드하여 `datasets` 라이브러리의 `Dataset` 객체로 변환했습니다.
- **데이터 포맷팅**: 모델 학습을 위해 각 데이터를 아래와 같은 프롬프트 형식으로 가공했습니다.

  ```python
  # 데이터 포맷팅에 사용된 실제 코드
  def formatting_prompts_func(example):
      text = f"""### Instruction:
Below is a text from a research paper on early childhood. Summarize the key findings or main points of the following text.

### Input:
{example['text']}

### Response:
{example['summary']}"""
      return [text]

#### 다. 모델 학습 (Fine-tuning)

  - **PEFT 설정**: `peft` 라이브러리의 `LoraConfig`를 사용하여 LoRA(Low-Rank Adaptation) 파라미터를 설정했습니다. 이를 통해 전체 모델 파라미터를 업데이트하지 않고, 일부 어댑터만 학습하여 효율성을 극대화했습니다.
  - **학습**: `SFTTrainer`를 사용하여 모델 학습을 진행했습니다. 주요 학습 파라미터는 다음과 같습니다.
      - `max_seq_length`: 2048
      - `per_device_train_batch_size`: 2
      - `gradient_accumulation_steps`: 4
      - `optimizer`: "adamw\_8bit"

<br>

## 4\. WBS
<img width="1319" height="665" alt="image" src="https://github.com/user-attachments/assets/c322baca-e38f-40ce-b0d8-a4234f4a4f97" />


<br>

## 5\. 데이터 분석 및 시각화

> 💡 **팁**: Jupyter Notebook에서 시각화한 결과(그래프, 표 등)를 이미지 파일로 저장한 후, 아래와 같이 README에 추가하여 프로젝트 결과를 효과적으로 보여줄 수 있습니다.

#### 가. 데이터셋 키워드 분포

학습에 사용된 `output.jsonl` 데이터셋의 주요 키워드를 분석한 결과, '유아', '상호작용', '사회성', '발달', '애착' 등의 단어가 높은 빈도로 나타났습니다. 이는 우리 모델이 부모와 자녀 간의 관계 및 유아의 핵심 발달 과업에 대한 지식을 집중적으로 학습했음을 시사합니다.

`![키워드 분포](images/keyword_distribution.png)`

#### 나. 모델 학습 결과 (Loss)

모델 학습 과정에서의 Training Loss는 꾸준히 감소하여 안정적으로 수렴하는 양상을 보였습니다. 이는 모델이 학습 데이터를 효과적으로 학습하고 있으며, 과적합(overfitting) 없이 파인튜닝이 성공적으로 진행되었음을 나타냅니다.

`![Training Loss](images/training_loss.png)`

<br>

## 6\. 기대 효과 및 향후 과제

  - **기대 효과**: 육아 분야에 특화된 질의응답이 가능한 AI 모델을 개발하여, 부모나 교사에게 신뢰도 높은 정보를 제공할 수 있을 것으로 기대됩니다. 사용자는 검증된 학술 정보를 바탕으로 한 답변을 통해 자녀 양육에 대한 확신과 전문성을 높일 수 있습니다.
  - **향후 과제**:
      - 더 많은 양질의 데이터 확보 및 정제
      - 하이퍼파라미터 튜닝을 통한 모델 성능 최적화
      - 개발된 모델을 활용한 실제 적용 사례(예: 챗봇 서비스, 육아 정보 요약 서비스) 개발
      - RAG(Retrieval-Augmented Generation) 기술을 도입하여 최신 논문 정보를 실시간으로 반영하는 시스템 구축

<br>

## 7\. 한줄 회고록

  - **김민균**: QLoRA와 `unsloth`를 활용해 제한된 GPU 환경에서도 대규모 모델을 효율적으로 파인튜닝할 수 있다는 점이 인상 깊었습니다.
  - **김세한**: 신뢰도 높은 데이터셋을 구축하기 위해 논문을 정제하는 과정이 까다로웠지만, 모델 성능의 기반이 된다는 점에서 큰 보람을 느꼈습니다.
  - **김수현**: 웹 크롤링을 통해 데이터를 수집하는 과정에서 스스로의 부족한 점을 돌아보게 되었습니다. 일부 데이터를 수집했는데 막상 분석하고 활용하기에는 어려운 데이터가 많아 아쉬움이 남았습니다. 이는 저의 경험 부족에서 비롯된 것이라 생각하며, 앞으로는 데이터의 품질과 활용성을 높일 수 있는 방법에 대해 더 깊이 공부하고 노력하는 자세를 갖겠습니다.
  - **정의중**: 명확한 목표를 공유하고 각자의 역할에 최선을 다한 덕분에 시너지를 낼 수 있었습니다. 최고의 팀워크였습니다\!
  - **최우진**: LLM 파인튜닝의 전 과정을 직접 경험하며 이론으로만 알던 개념들을 체득할 수 있었던 값진 프로젝트였습니다.

