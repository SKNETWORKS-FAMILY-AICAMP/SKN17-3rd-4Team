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

- **프로젝트 소개**:
- **프로젝트 필요성**:
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
- **모델**: 허깅페이스(Hugging Face)에 공개된 `unsloth/Qwen2.5-14B` 모델을 사용했습니다.
- **모델 선정 기준**
  - 0
  - 0
  - 0
  - 0
- **양자화 (Quantization)**: 메모리 부족 문제(`RuntimeError`)를 해결하고 효율적인 학습을 위해 `BitsAndBytesConfig`를 사용하여 4-bit QLoRA 양자화를 적용했습니다.

#### 나. 데이터 전처리
- **데이터 로드**: `output.jsonl` 파일을 로드하여 `datasets` 라이브러리의 `Dataset` 객체로 변환했습니다.
- **데이터 포맷팅**: 모델 학습을 위해 각 데이터를 아래와 같은 프롬프트 형식으로 가공했습니다.

  ```python
  # 데이터 포맷팅에 사용된 실제 코드를 여기에 붙여넣으세요.
  def formatting_prompts_func(example):
      text = f"""### Instruction:
Below is a text from a research paper on early childhood. Summarize the key findings or main points of the following text.

### Input:
{example['text']}

### Response:
{example['summary']}

#### 다. 모델 학습 (Fine-tuning)

  - **PEFT 설정**: `peft` 라이브러리의 `LoraConfig`를 사용하여 LoRA(Low-Rank Adaptation) 파라미터를 설정했습니다.
  - **학습**: `SFTTrainer`를 사용하여 모델 학습을 진행했습니다. 주요 학습 파라미터는 다음과 같습니다.
      - `max_seq_length`: 2048
      - `per_device_train_batch_size`: 2
      - `gradient_accumulation_steps`: 4
      - `optimizer`: "adamw\_8bit"

<br>

## 4\. 작업 계획

| 단계 | 작업 내용 | 시작일 | 종료일 | 비고 |
|:---:|:---|:---:|:---:|:---|
| 1 | 데이터 수집 및 정제 | 2025-09-20 | 2025-09-21 | 육아 관련 논문 텍스트 추출 |
| 2 | 모델 선정 및 환경 구축 | 2025-09-21 | 2025-09-21 | `unsloth/Qwen2.5-14B` 선정 |
| 3 | 데이터 전처리 및 포맷팅 | 2025-09-22 | 2025-09-23 | `datasets` 라이브러리 활용 |
| 4 | **모델 파인튜닝** | 2025-09-23 | 2025-09-24 | **(현재 진행 중)** |
| 5 | 모델 성능 평가 및 검증 | - | - | |
| 6 | 결과 분석 및 모델 활용 | - | - | |

<br>

## 5\. 데이터 분석 및 시각화

> 💡 **팁**: Jupyter Notebook에서 시각화한 결과(그래프, 표 등)를 이미지 파일로 저장한 후, 아래와 같이 README에 추가하여 프로젝트 결과를 효과적으로 보여줄 수 있습니다.

#### 가. 데이터셋 키워드 분포

[여기에 `output.jsonl` 데이터의 카테고리나 키워드 분포를 시각화한 그래프 이미지를 넣어주세요. 예: "유아", "수줍음", "사회성"]

`![키워드 분포](images/keyword_distribution.png)`

#### 나. 모델 학습 결과 (Loss)

[여기에 모델 학습 과정에서 출력된 Loss 그래프 이미지를 넣어주세요.]

`![Training Loss](images/training_loss.png)`

<br>

## 6\. 기대 효과 및 향후 과제

  - **기대 효과**: 육아 분야에 특화된 질의응답이 가능한 AI 모델을 개발하여, 부모나 교사에게 신뢰도 높은 정보를 제공할 수 있을 것으로 기대됩니다.
  - **향후 과제**:
      - 더 많은 양질의 데이터 확보 및 정제
      - 하이퍼파라미터 튜닝을 통한 모델 성능 최적화
      - 개발된 모델을 활용한 실제 적용 사례(예: 챗봇 서비스) 개발

<br>

## 7\. 한줄 회고록

  - **김민균**:
  - **김세한**:
  - **김수현**:
  - **정의중**:
  - **최우진**:
