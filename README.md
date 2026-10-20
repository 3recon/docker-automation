Docker Desktop RPA 봇 생성기 · SLM 학습 데이터 (POC)
사용자의 자연어 요청을 "Docker Desktop에서 실행할 작업 계획 JSON"으로 바꾸는 SLM을 학습시키기 위한 도구입니다.
`wag_docker.py` 파일 하나에 설정과 코드가 모두 들어 있고, 학습 데이터는 이 파일을 실행해서 만듭니다.
SLM은 이해만 담당합니다. 발화를 읽고 계획을 갱신하며, 모르는 값은 비워 둡니다.
질문을 할지, 생성 [예]/[아니오] 버튼을 띄울지는 규칙 코드인 `decide()`가 정합니다. 그래서 학습 데이터는 SLM 입력→출력 한 종류뿐입니다.
```
사용자 발화 ─▶ SLM (계획 갱신) ─▶ decide() ─┬─▶ ask      : 질문 + 선택지 버튼
                    ▲                        ├─▶ confirm  : 실행 계획 + [예]/[아니오]
                    └── 다음 턴 입력(현재 계획, 직전 질문) ◀┘─▶ reject / cancelled
```
빠른 시작
```bash
python wag_docker.py gen        # data/ 폴더에 학습·평가 데이터 생성 + 자동 검증
```
결과:
파일	내용
`data/train_sft.jsonl`	학습용 1,200개 (`messages` 형식)
`data/eval_sft.jsonl`	평가용 250개 (`messages` 형식)
`data/train.jsonl`, `data/eval.jsonl`	원본 레코드 (환경, 케이스 종류, 챗봇 다음 행동 포함). 채점에 사용
`data/stats.json`	케이스별 개수
시드(기본 42)가 같으면 언제 어디서 돌려도 똑같은 데이터가 나옵니다 (Python 3.11~3.13에서 확인). 마지막에 `0 errors`가 두 번 나오면 정상입니다.
다른 명령:
```bash
python wag_docker.py gen --train 2000 --eval 300 --seed 7 --out data2   # 크기·시드 바꾸기
python wag_docker.py validate data                                      # 검증만 다시
python wag_docker.py preview data/train.jsonl 40 > preview.md          # 사람이 읽는 미리보기
python wag_docker.py score data/eval.jsonl preds.jsonl                 # 모델 예측 채점
```
코랩에서 학습하기
```python
# 1) 저장소 받기 (공개 저장소)
!git clone https://github.com/3recon/docker-automation.git
%cd docker-automation

# 비공개 저장소라면: 왼쪽 열쇠 아이콘(Secrets)에 GITHUB_TOKEN 저장 후
# from google.colab import userdata
# token = userdata.get('GITHUB_TOKEN')
# !git clone https://{token}@github.com/3recon/docker-automation.git

# 2) 데이터 만들기 (몇 초)
!python wag_docker.py gen

# 3) 불러오기
from datasets import load_dataset
ds = load_dataset("json", data_files={"train": "data/train_sft.jsonl", "eval": "data/eval_sft.jsonl"})
```
`messages` 형식이라 TRL `SFTTrainer`, Unsloth, Axolotl, LLaMA-Factory에 그대로 넣으면 됩니다.
POC 권장 설정:
모델: 한국어가 되는 1.5B~4B instruct 모델. 무료 T4라면 1.5B는 LoRA, 3~4B는 4비트 QLoRA
LoRA r=16, alpha=32, lr 2e-4, 2~3 epoch, max_seq_len 2048
assistant 토큰만 loss 계산 (TRL: `assistant_only_loss=True` 또는 completion-only collator). 입력이 출력보다 훨씬 길어서 중요합니다.
런타임이 끊기면 결과가 사라지므로 LoRA 어댑터는 Google Drive나 Hugging Face Hub에 저장하세요.
예시 하나가 문자 기준 중앙값 약 1,960자이며, 그중 시스템 프롬프트가 약 1,150자입니다.
평가
평가셋을 모델로 예측해 `{"id": ..., "prediction": "모델 출력 원문"}` 형식의 `preds.jsonl`을 만든 뒤 채점합니다.
```python
import json
# model, tok = 학습한 모델과 토크나이저
with open("preds.jsonl", "w", encoding="utf-8") as f:
    for line in open("data/eval_sft.jsonl", encoding="utf-8"):
        row = json.loads(line)
        prompt = tok.apply_chat_template(row["messages"][:2], tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        f.write(json.dumps({"id": row["id"], "prediction": text}, ensure_ascii=False) + "\n")
```
```bash
!python wag_docker.py score data/eval.jsonl preds.jsonl
```
가장 중요한 지표는 `behavior`입니다. 예측을 `decide()`에 넣었을 때 챗봇이 정답과 똑같이 행동(같은 질문·같은 선택지·같은 실행 계획)했는지를 봅니다. `unseen_pattern` 행으로 처음 보는 문장에 대한 일반화도 따로 확인할 수 있습니다. 틀린 예시는 `preds_fails.jsonl`로 저장됩니다.
vLLM 같은 추론 서버에 올릴 때는 `build_schema(ctx, SPEC)`로 현재 환경의 이름을 enum으로 채운 JSON Schema를 만들어 제약 디코딩을 걸 수 있습니다. 환경에 없는 이름이나 없는 작업은 아예 생성되지 않습니다.
지원 작업
작업	대상	전제 조건 (정책이 확인)	되돌릴 수 없음
`container.start`	컨테이너	중지 상태여야 함	
`container.stop`	컨테이너	실행 중이어야 함	
`container.restart`	컨테이너		
`container.delete`	컨테이너		✔
`container.logs`	컨테이너		
`image.run` (새 컨테이너 생성·실행)	이미지		
`image.delete`	이미지	사용하는 컨테이너가 없어야 함	✔
`image.prune`	없음		✔
`volume.delete`	볼륨	사용하는 컨테이너가 없어야 함	✔
SLM 입력과 출력
입력 (user 메시지):
```
[환경]
컨테이너:
- shop-nginx-1 (nginx:1.27)
- shop-redis-1 (redis:7)
- blog-redis-1 (redis:7.2)
이미지:
- nginx:1.27
- redis:7
- redis:7.2
볼륨:
- shop_redis-data
- blog_redis-data
[현재 계획]
없음
[직전 질문]
없음
[사용자]
레디스 좀 재시작해줘
```
출력 (assistant 메시지):
```json
{"tasks":[{"task_candidates":["container.restart"],"args":{"container":{"mention":"레디스","candidates":["shop-redis-1","blog-redis-1"]}}}],"unsupported":[],"cancel":false}
```
후보가 2개이므로 정책이 "'레디스'에 해당하는 컨테이너가 여러 개예요. 어느 것인가요? [shop-redis-1] [blog-redis-1]"이라고 묻습니다.
사용자가 "2번"이라고 답하면 다음 턴 입력의 `[현재 계획]`에 위 계획이, `[직전 질문]`에 선택지가 들어가고, SLM은 `candidates: ["blog-redis-1"]`로 고친 전체 계획을 출력합니다.
출력 규칙 요약:
`candidates`에는 환경에 실제로 있는 이름만 넣습니다. 표현에 해당하는 것을 모두 넣고, 없으면 `[]`.
대상을 말하지 않으면 `mention: null`, `candidates: []`.
작업이 애매하면 `task_candidates`에 가능한 작업을 모두 넣습니다 ("DB 정리해줘" → 중지/삭제).
실행 상태·사용 여부는 SLM이 따지지 않습니다. 환경 입력에도 넣지 않았고, 정책이 거릅니다.
표현이 어떤 항목의 이름과 똑같으면 그 항목 하나만 가리킵니다 (컨테이너 `postgres`와 `my-postgres`가 있을 때 "postgres" → `postgres`만).
같은 표현이 여러 서비스에 걸리면 모두 후보입니다. "웹"은 web 앱(별칭)과 nginx(역할 이름)를 모두 가리키므로, 둘 다 있으면 정책이 어느 쪽인지 묻습니다. 역할 이름("웹서버", "DB", "캐시")은 컨테이너·이미지·볼륨 모두에 적용됩니다.
"만들고 실행", "새로 띄워", "하나 더 만들어"는 `image.run` 하나입니다. 이미 있는 컨테이너를 켜는 "실행해줘", "켜줘"는 `container.start`입니다.
케이스 구성 (기본 설정 기준)
케이스	예시 발화	train	eval
대상이 하나로 정해짐	`wiki 레디스 꺼줘`, `캐시 서버 재시작`	259	41
대상을 말하지 않음	`컨테이너 하나 꺼줘`, `새 컨테이너 하나 띄워줘`	159	19
대상이 여러 개	`데이터베이스 지워버려` (DB가 2개)	159	31
대상이 환경에 없음	`MQ가 응답이 없어 재시작 좀`	83	22
작업이 애매함	`postgres가 이상해` (재시작? 로그?)	121	35
서로 다른 작업 2개	`chat_redis 날리고 안 쓰는 이미지 정리해줘`	125	29
같은 작업, 대상 2개	`대시보드하고 빌드 서버 중지해줘`	48	6
같은 대상, 작업 2개	`래빗엠큐 실행하고 로그 확인해줘`	39	12
이미지 정리 (대상 없음)	`필요 없는 이미지 싹 정리해줘`	23	13
범위 밖 요청	`도커 로그인해줘`, `X 포트를 8080으로 바꿔줘`	69	13
지원 + 범위 밖 섞임	`pg를 켜고 메모리 제한 걸어줘`	52	9
새로 만들기 vs 켜기 대조	`웹 컨테이너 생성해줘` ↔ `웹 컨테이너 실행 부탁해요`	63	20
위 숫자는 각 케이스에서 이어진 후속 턴까지 포함한 개수입니다. 후속 턴은 질문에 대한 답변("2번", "프론트엔드요", "로그 확인"), 확인 화면에서 [아니오] 뒤의 수정("redis 말고 몽고로 해줘", "그리고 그거 로그도 보여줘", "admin은 빼줘"), 취소("그냥 취소할게")입니다. train 1,200개 중 369개, eval 250개 중 73개가 후속 턴입니다.
같은 대상을 공유하는 작업("웹 로그 보여주고 재시작해줘")에서 대상을 물었을 때, 사용자의 답은 그 대상을 쓰는 모든 작업에 함께 적용됩니다. 같은 질문을 두 번 하지 않습니다.
정답 후보는 사람이 라벨링하지 않고 가상 환경 정보로 계산합니다. `validate`는 생성기와 따로 작성한 해석기로 후보를 다시 계산해 비교합니다.
평가셋 분리
프로젝트 이름(shop, blog... / crm, lms, booking, hr)과 단독 컨테이너 이름 형식이 train과 겹치지 않습니다.
각 패턴 목록에서 5번째마다의 문장은 평가 전용입니다. eval 250개 중 143개가 학습에 없던 문장 패턴을 씁니다 (`unseen_pattern: true`).
환경과 발화가 모두 같은 예시는 train과 eval 사이에 없습니다. "2번" 같은 짧은 답변 문자열만 겹칩니다.
작업·문장 추가
`wag_docker.py` 맨 위 "1. 설정" 부분만 고치면 됩니다.
`SPEC`에 작업 템플릿 추가
`SEEDS["tasks"]`에 같은 키로 문장 패턴(`final`, `final_noslot`, `conn`, `add`) 추가
`SYSTEM_PROMPT`의 작업 목록에 한 줄 추가
`python wag_docker.py gen`으로 다시 생성 (오류 0건 확인)
실서비스 연결
docker CLI로 환경을 만듭니다.
컨테이너: `docker ps -a --format "{{.Names}}\t{{.Image}}\t{{.State}}"`
이미지: `docker images --format "{{.Repository}}:{{.Tag}}"` (사용 여부는 컨테이너의 이미지 목록과 대조)
볼륨: `docker volume ls --format "{{.Name}}"` (사용 여부는 `docker ps -a --filter volume=<이름> -q`)
`build_messages()`로 입력을 만들고 `build_schema()`로 제약 디코딩을 겁니다.
출력을 `decide()`에 넣어 질문·확인·거절을 정합니다.
사용자가 선택지 버튼을 누르면 SLM 없이 코드가 `pending` 위치의 후보를 바로 채웁니다. SLM은 사용자가 직접 타이핑한 답변에만 씁니다.
[예]를 누르면 계획의 각 작업을 beSRT 시나리오(버튼 클릭 순서)로 펼쳐 실행합니다. (변환기는 아직 없음)
`import wag_docker`로 불러 쓰면 됩니다.
한계와 다음 단계
문장이 템플릿 조합이라 실제 사용자 표현보다 다양성이 낮습니다. POC에서 가능성이 보이면 큰 LLM으로 시드 문장을 패러프레이즈하고 실제 사용 로그를 검수해 추가하세요.
"실행 중인 거 다 꺼줘" 같은 일괄 작업은 범위에서 뺐습니다.
확인 단계의 "아니오 → 대상 제거" 예시는 적습니다 (train 4건).
환경에 없는 이미지로 새 컨테이너 만들기(먼저 pull 필요)와 포트·이름 지정("8080 포트로 웹 컨테이너 만들어줘")은 범위 밖입니다. 다음 단계 후보는 `image.pull` 작업과 `image.run`의 `port` 슬롯입니다.
