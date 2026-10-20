#!/usr/bin/env python3
"""Docker Desktop RPA 봇 생성기 - SLM 학습 데이터 도구 (단일 파일)

자연어 요청을 "Docker Desktop 작업 계획 JSON"으로 바꾸는 SLM의 학습 데이터를 만들고, 검증하고, 채점한다.
설정(작업·사전·문장 패턴·시스템 프롬프트)과 코드가 모두 이 파일 안에 있다.

사용법
  python wag_docker.py gen                                  # data/ 에 train 1,200 / eval 250 생성 + 자동 검증
  python wag_docker.py gen --train 2000 --eval 300 --seed 7 --out data2
  python wag_docker.py validate data                        # 데이터 검증 (오류 0건이어야 함)
  python wag_docker.py preview data/train.jsonl 30          # 사람이 읽기 좋은 미리보기 (마크다운)
  python wag_docker.py score data/eval.jsonl preds.jsonl    # 모델 예측 채점

파일 구성 (위에서부터)
  1. 설정: SPEC(작업 템플릿), LEX(서비스 사전), SEEDS(발화 패턴), SYSTEM_PROMPT
  2. 조사 처리: attach()
  3. 대화 정책: decide()  - 질문 / 확인 버튼 / 거절 결정. 실서비스에서도 그대로 쓴다
  4. 프롬프트 조립: build_messages(), build_schema()
  5. 데이터 생성기: Gen, cmd_gen()
  6. 검증·채점·미리보기: cmd_validate(), cmd_score(), cmd_preview()
  7. 명령줄 진입점: main()
"""
import argparse
import copy
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

# ============================================================================
# 1. 설정 (작업을 추가하거나 문장 패턴을 늘릴 때 여기를 고친다)
# ============================================================================

# 작업 템플릿: 대상 종류(slot), 전제 조건, 되돌릴 수 없는지, 질문·계획 문구
SPEC = json.loads(r'''
{
    "container.start": {"label": "시작", "slot": "container", "destructive": false, "precondition": "not_running", "question": "어떤 컨테이너를 시작할까요?", "no_option": "시작할 수 있는 컨테이너가 없어요.", "plan": "{target} 컨테이너 시작"},
    "container.stop": {"label": "중지", "slot": "container", "destructive": false, "precondition": "running", "question": "어떤 컨테이너를 중지할까요?", "no_option": "중지할 수 있는 실행 중인 컨테이너가 없어요.", "plan": "{target} 컨테이너 중지"},
    "container.restart": {"label": "재시작", "slot": "container", "destructive": false, "precondition": null, "question": "어떤 컨테이너를 재시작할까요?", "no_option": "재시작할 컨테이너가 없어요.", "plan": "{target} 컨테이너 재시작"},
    "container.delete": {"label": "삭제", "slot": "container", "destructive": true, "precondition": null, "question": "어떤 컨테이너를 삭제할까요?", "no_option": "삭제할 컨테이너가 없어요.", "plan": "{target} 컨테이너 삭제"},
    "container.logs": {"label": "로그 보기", "slot": "container", "destructive": false, "precondition": null, "question": "어떤 컨테이너의 로그를 볼까요?", "no_option": "로그를 볼 컨테이너가 없어요.", "plan": "{target} 컨테이너 로그 열기"},
    "image.run": {"label": "새 컨테이너 생성", "slot": "image", "destructive": false, "precondition": null, "question": "어떤 이미지로 새 컨테이너를 만들까요?", "no_option": "실행할 이미지가 없어요.", "plan": "{target} 이미지로 새 컨테이너 실행"},
    "image.delete": {"label": "이미지 삭제", "slot": "image", "destructive": true, "precondition": "not_in_use", "question": "어떤 이미지를 삭제할까요?", "no_option": "삭제할 수 있는(사용 중이 아닌) 이미지가 없어요.", "plan": "{target} 이미지 삭제"},
    "image.prune": {"label": "안 쓰는 이미지 정리", "slot": null, "destructive": true, "precondition": null, "question": null, "no_option": null, "plan": "사용하지 않는 이미지 정리"},
    "volume.delete": {"label": "볼륨 삭제", "slot": "volume", "destructive": true, "precondition": "not_in_use", "question": "어떤 볼륨을 삭제할까요?", "no_option": "삭제할 수 있는(사용 중이 아닌) 볼륨이 없어요.", "plan": "{target} 볼륨 삭제"}
}
''')

# 가상 Docker 환경용 서비스 사전
LEX = json.loads(r'''
{
    "_comment": "가상 Docker 환경을 무작위로 만들 때 쓰는 서비스 사전. aliases = 사용자가 부르는 이름, categories = 역할로 부르는 이름, volume = 상태 저장용 볼륨 접미사 | 같은 표현이 여러 서비스에 걸리면(예: '웹' = web 앱의 별칭 + nginx의 역할 이름) 모두 후보가 된다",
    "infra_services": {
        "nginx": {"repo": "nginx", "tags": ["1.27", "latest", "alpine", "1.25"], "aliases": ["nginx", "엔진엑스"], "categories": ["웹서버", "리버스 프록시", "웹"], "volume": null},
        "redis": {"repo": "redis", "tags": ["7", "7.2", "alpine"], "aliases": ["redis", "레디스"], "categories": ["캐시", "캐시 서버"], "volume": "redis-data"},
        "postgres": {"repo": "postgres", "tags": ["16", "15"], "aliases": ["postgres", "포스트그레스", "pg"], "categories": ["DB", "디비", "데이터베이스"], "volume": "pgdata"},
        "mysql": {"repo": "mysql", "tags": ["8.0", "8.4"], "aliases": ["mysql", "마이에스큐엘"], "categories": ["DB", "디비", "데이터베이스"], "volume": "mysql-data"},
        "mongo": {"repo": "mongo", "tags": ["7", "6"], "aliases": ["mongo", "몽고", "몽고디비"], "categories": ["DB", "디비", "데이터베이스"], "volume": "mongo-data"},
        "rabbitmq": {"repo": "rabbitmq", "tags": ["3-management", "3.13"], "aliases": ["rabbitmq", "래빗엠큐"], "categories": ["메시지 큐", "MQ"], "volume": null},
        "kafka": {"repo": "bitnami/kafka", "tags": ["3.7", "latest"], "aliases": ["kafka", "카프카"], "categories": ["메시지 큐", "MQ"], "volume": "kafka-data"},
        "elasticsearch": {"repo": "elasticsearch", "tags": ["8.15.0", "8.14.3"], "aliases": ["elasticsearch", "엘라스틱서치", "엘라스틱"], "categories": ["검색 엔진"], "volume": "esdata"},
        "grafana": {"repo": "grafana/grafana", "tags": ["11.2.0", "latest"], "aliases": ["grafana", "그라파나"], "categories": ["모니터링", "대시보드"], "volume": "grafana-data"},
        "prometheus": {"repo": "prom/prometheus", "tags": ["v2.54.1", "latest"], "aliases": ["prometheus", "프로메테우스"], "categories": ["모니터링"], "volume": null},
        "jenkins": {"repo": "jenkins/jenkins", "tags": ["lts", "2.462"], "aliases": ["jenkins", "젠킨스"], "categories": ["CI 서버", "빌드 서버"], "volume": "jenkins_home"},
        "minio": {"repo": "minio/minio", "tags": ["latest"], "aliases": ["minio", "미니오"], "categories": ["오브젝트 스토리지"], "volume": "minio-data"}
    },
    "app_services": {
        "api": {"tags": ["latest", "dev"], "aliases": ["api", "api 서버", "백엔드"], "categories": []},
        "web": {"tags": ["latest", "dev"], "aliases": ["web", "웹", "프론트", "프론트엔드"], "categories": []},
        "worker": {"tags": ["latest"], "aliases": ["worker", "워커"], "categories": []},
        "admin": {"tags": ["latest"], "aliases": ["admin", "어드민"], "categories": []}
    },
    "projects": {
        "train": ["shop", "blog", "myapp", "erp", "chat", "pay", "wiki", "mall"],
        "eval": ["crm", "lms", "booking", "hr"]
    },
    "standalone_name_forms": {
        "train": ["{s}", "my-{s}", "{s}-test"],
        "eval": ["{s}", "local-{s}", "{s}-dev"]
    },
    "extra_images": ["node:20", "python:3.12", "ubuntu:24.04", "hello-world:latest", "busybox:latest", "alpine:3.20"],
    "orphan_volume_prefixes": {
        "train": ["old", "test"],
        "eval": ["backup", "tmp"]
    }
}
''')

# 발화 패턴
SEEDS = json.loads(r'''
{
    "_comment": "{C}=컨테이너 표현, {I}=이미지 표현, {V}=볼륨 표현. {C|을}처럼 쓰면 받침에 맞춰 을/를이 붙는다. 각 목록에서 5번째마다(인덱스 4, 9, 14...) 패턴은 평가셋 전용이다.",
    "tasks": {
        "container.start": {
            "final": ["{C|을} 켜줘", "{C} 실행해줘", "{C|을} 시작해 줘", "{C} 좀 띄워줘", "{C} 컨테이너 시작", "{C} 올려", "{C} 컨테이너 실행 부탁해요", "{C|을} 구동해줘", "start {C}", "{C} 기동시켜 주세요", "{C|이} 꺼져 있던데 켜줘", "{C} 좀 올려줄래?", "{C|을} 실행시켜 주세요", "{C} 켜", "{C} 컨테이너 좀 살려줘"],
            "final_noslot": ["컨테이너 하나 켜줘", "컨테이너 시작해줘", "컨테이너 좀 실행해 줄래?", "컨테이너 하나 띄워줘", "꺼둔 컨테이너 하나 켜줘"],
            "conn": ["{C|을} 켜고", "{C} 실행하고", "{C|을} 시작하고", "{C} 띄우고", "{C} 올리고"],
            "implicit": [],
            "add": ["{C|도} 켜줘", "그리고 {C|도} 시작해줘", "{C|도} 같이 띄워줘"]
        },
        "container.stop": {
            "final": ["{C|을} 꺼줘", "{C} 중지해줘", "{C|을} 멈춰줘", "{C} 좀 내려줘", "{C} 컨테이너 stop", "{C} 정지시켜 주세요", "{C|을} 종료해줘", "{C} 내려", "stop {C}", "{C} 잠깐 꺼둘래", "{C|이} 메모리 많이 먹네 꺼줘", "{C} 중단해 주세요", "{C} 컨테이너 꺼", "{C} 멈춰", "{C|을} 정지해 줄래?"],
            "final_noslot": ["컨테이너 하나 꺼줘", "실행 중인 컨테이너 하나 멈춰줘", "컨테이너 좀 내려줘", "컨테이너 중지해줘", "돌고 있는 컨테이너 하나 꺼줘"],
            "conn": ["{C|을} 끄고", "{C} 중지하고", "{C|을} 멈추고", "{C} 내리고", "{C} 정지하고"],
            "implicit": [],
            "add": ["{C|도} 같이 꺼줘", "그리고 {C|도} 중지해줘", "{C|도} 내려줘"]
        },
        "container.restart": {
            "final": ["{C|을} 재시작해줘", "{C} 리스타트", "{C} 다시 시작해줘", "{C|을} 껐다 켜줘", "{C} 재기동해 주세요", "restart {C}", "{C|이} 응답이 없어 재시작 좀", "{C} 한번 재부팅해줘", "{C} 리스타트 부탁해요", "{C} 리부트해줘", "{C|을} 다시 띄워줘", "{C} 재시작", "{C|이} 이상하게 느려 재시작해줘", "{C} 껐다가 다시 켜", "{C} 컨테이너 재시작 좀 해줄래?"],
            "final_noslot": ["컨테이너 재시작해줘", "컨테이너 하나 껐다 켜줘", "컨테이너 리스타트 해줘", "컨테이너 다시 시작해줘", "컨테이너 하나 재기동해줘"],
            "conn": ["{C|을} 재시작하고", "{C} 리스타트하고", "{C|을} 껐다 켜고", "{C} 다시 시작하고", "{C} 재기동하고"],
            "implicit": ["재시작해줘", "다시 시작해줘", "껐다 켜줘"],
            "add": ["{C|도} 재시작해줘", "그리고 {C|도} 리스타트해줘", "{C|도} 같이 껐다 켜줘"]
        },
        "container.delete": {
            "final": ["{C|을} 삭제해줘", "{C} 지워줘", "{C} 컨테이너 제거해 주세요", "{C|은} 이제 안 써 삭제해", "rm {C}", "{C|을} 없애줘", "{C} 컨테이너 날려줘", "{C} 삭제 부탁드립니다", "delete {C}", "{C} 컨테이너 지워", "{C|은} 필요 없으니까 지워줘", "{C|을} 제거해줘", "{C} 컨테이너 삭제", "{C} 지워버려", "{C|을} 삭제해 줄래?"],
            "final_noslot": ["컨테이너 하나 지워줘", "컨테이너 삭제해줘", "안 쓰는 컨테이너 하나 지워줘", "컨테이너 제거해 주세요", "컨테이너 하나 날려줘"],
            "conn": ["{C|을} 삭제하고", "{C} 지우고", "{C|을} 제거하고", "{C} 날리고", "{C|을} 없애고"],
            "implicit": ["지워줘", "삭제해줘", "제거해줘"],
            "add": ["{C|도} 지워줘", "그리고 {C|도} 삭제해줘", "{C|도} 같이 제거해줘"]
        },
        "container.logs": {
            "final": ["{C} 로그 보여줘", "{C} 로그 좀 볼게", "{C} 로그 확인해줘", "{C|이} 왜 죽었는지 로그 보여줘", "{C} 컨테이너 로그 열어줘", "logs {C}", "{C} 에러 로그 확인", "{C} 로그 띄워 주세요", "{C} 로그 볼 수 있어?", "{C} 로그 열어", "{C} 컨테이너 로그 보고 싶어", "{C} 로그 체크해줘", "{C|이} 에러 나는 것 같은데 로그 보여줘", "{C} 로그", "{C} 로그 화면 열어줘"],
            "final_noslot": ["로그 보여줘", "컨테이너 로그 좀 보자", "로그 확인해줘", "컨테이너 로그 열어줘", "로그 좀 볼게"],
            "conn": ["{C} 로그 보여주고", "{C} 로그 확인하고", "{C} 로그 열고", "{C} 로그 보고", "{C} 로그 띄우고"],
            "implicit": ["로그 보여줘", "로그 확인해줘", "로그 열어줘"],
            "add": ["{C} 로그도 보여줘", "그리고 {C} 로그도 열어줘", "{C} 로그도 확인해줘"]
        },
        "image.run": {
            "final": ["{I} 이미지로 컨테이너 하나 띄워줘", "{I} 이미지 실행해줘", "{I} 이미지 run 해줘", "{I} 이미지로 새 컨테이너 만들어줘", "{I} 이미지 돌려줘", "{I} 이미지로 컨테이너 실행", "{I} 이미지 하나 띄워", "{I} 이미지로 새로 하나 실행해 주세요", "run {I}", "{I} 이미지 컨테이너로 올려줘", "{I} 컨테이너 만들고 실행해줘", "새 {I} 컨테이너 띄워줘", "{I} 컨테이너 하나 더 만들어줘", "{I} 컨테이너 새로 만들어서 실행해줘", "{I} 컨테이너 생성해서 실행해 주세요", "{I|으로} 컨테이너 하나 새로 띄워줘", "{I} 컨테이너 하나 만들어줘", "새 {I} 컨테이너 하나 실행해줘", "{I} 컨테이너 한 개 더 띄워줘", "{I} 컨테이너 새로 하나 만들어 주세요", "{I} 컨테이너 생성해줘", "{I} 새로 하나 띄워줘"],
            "final_noslot": ["이미지 하나 실행해줘", "이미지로 컨테이너 띄워줘", "이미지 run 해줘", "이미지로 새 컨테이너 만들어줘", "이미지 하나 돌려줘", "컨테이너 하나 새로 만들어줘", "새 컨테이너 만들어서 실행해줘", "컨테이너 하나 생성해줘", "새 컨테이너 하나 띄워줘", "컨테이너 하나 더 만들어줘"],
            "conn": ["{I} 이미지로 컨테이너 띄우고", "{I} 이미지 실행하고", "{I} 이미지 돌리고", "{I} 이미지로 새 컨테이너 만들고", "{I} 이미지 run 하고", "{I} 컨테이너 하나 새로 만들고", "새 {I} 컨테이너 띄우고"],
            "implicit": [],
            "add": ["{I} 이미지도 실행해줘", "그리고 {I} 이미지로도 하나 띄워줘"]
        },
        "image.delete": {
            "final": ["{I} 이미지 삭제해줘", "{I} 이미지 지워줘", "{I} 이미지는 필요 없어 지워", "rmi {I}", "{I} 이미지 제거해 주세요", "{I} 이미지 날려줘", "{I} 이미지 삭제", "{I} 이미지 좀 지워줄래?", "{I} 이미지 없애줘", "{I} 이미지는 안 쓰니까 삭제해줘"],
            "final_noslot": ["이미지 하나 지워줘", "이미지 삭제해줘", "이미지 하나 제거해줘", "이미지 좀 지워줘", "이미지 하나 날려줘"],
            "conn": ["{I} 이미지 삭제하고", "{I} 이미지 지우고", "{I} 이미지 제거하고", "{I} 이미지 날리고", "{I} 이미지 없애고"],
            "implicit": [],
            "add": ["{I} 이미지도 지워줘", "그리고 {I} 이미지도 삭제해줘"]
        },
        "image.prune": {
            "final": [],
            "final_noslot": ["안 쓰는 이미지 정리해줘", "사용하지 않는 이미지 다 지워줘", "이미지 정리 좀 해줘", "dangling 이미지 청소해줘", "용량 부족한데 안 쓰는 이미지 좀 비워줘", "docker image prune 해줘", "이미지 클린업 해줘", "필요 없는 이미지 싹 정리해줘", "쓰지 않는 이미지들 정리 부탁해요", "디스크 정리하게 미사용 이미지 지워줘"],
            "conn": ["안 쓰는 이미지 정리하고", "사용하지 않는 이미지 지우고", "이미지 정리하고", "미사용 이미지 비우고", "이미지 클린업 하고"],
            "implicit": [],
            "add": ["안 쓰는 이미지도 정리해줘", "그리고 이미지 정리도 해줘"]
        },
        "volume.delete": {
            "final": ["{V} 볼륨 삭제해줘", "{V} 볼륨 지워줘", "{V} 볼륨은 안 쓰니까 지워줘", "{V} 볼륨 제거 부탁해요", "volume rm {V}", "{V} 볼륨 날려줘", "{V} 볼륨 삭제", "{V} 볼륨 좀 없애줘", "{V} 볼륨 정리해줘", "{V} 볼륨 지워 주세요"],
            "final_noslot": ["볼륨 하나 지워줘", "볼륨 삭제해줘", "볼륨 하나 제거해줘", "볼륨 좀 지워줘", "볼륨 하나 날려줘"],
            "conn": ["{V} 볼륨 삭제하고", "{V} 볼륨 지우고", "{V} 볼륨 제거하고", "{V} 볼륨 날리고", "{V} 볼륨 없애고"],
            "implicit": [],
            "add": ["{V} 볼륨도 지워줘", "그리고 {V} 볼륨도 삭제해줘"]
        }
    },
    "implicit_pairs": [["container.logs", "container.restart"], ["container.restart", "container.logs"], ["container.stop", "container.delete"], ["container.start", "container.logs"]],
    "multi_same_joiners": ["{A|랑} {B}", "{A|과} {B}", "{A}, {B}", "{A|하고} {B}"],
    "ambiguous_task": [{"pattern": "{C} 정리해줘", "candidates": ["container.stop", "container.delete"]}, {"pattern": "{C} 치워줘", "candidates": ["container.stop", "container.delete"]}, {"pattern": "{C|이} 이상해", "candidates": ["container.restart", "container.logs"]}, {"pattern": "{C|이} 먹통이야", "candidates": ["container.restart", "container.logs"]}, {"pattern": "{C} 좀 처리해줘", "candidates": ["container.stop", "container.delete"]}, {"pattern": "{C|이} 안 돼", "candidates": ["container.restart", "container.logs"]}, {"pattern": "{C} 좀 봐줘", "candidates": ["container.restart", "container.logs"]}, {"pattern": "{C} 그만 돌려도 될 것 같아", "candidates": ["container.stop", "container.delete"]}, {"pattern": "{C|이} 계속 죽어", "candidates": ["container.restart", "container.logs"]}, {"pattern": "{C} 손 좀 봐줘", "candidates": ["container.restart", "container.logs"]}],
    "unsupported": ["쿠버네티스 클러스터 만들어줘", "도커 컴포즈 파일 작성해줘", "{C} 포트를 8080으로 바꿔줘", "새 이미지 빌드해줘", "도커 허브에 이미지 푸시해줘", "{C} 안에 들어가서 쉘 열어줘", "도커 데스크톱 업데이트해줘", "오늘 날씨 어때?", "{C} 환경변수 추가해줘", "네트워크 하나 새로 만들어줘", "점심 메뉴 추천해줘", "{C} 이름 바꿔줘", "깃 커밋해줘", "{C} CPU 사용량 그래프로 보여줘", "dockerfile 최적화해줘", "{C|을} 다른 서버로 옮겨줘", "도커 로그인해줘", "{C} 메모리 제한 2기가로 설정해줘", "안녕?", "{C} 설정 파일 수정해줘"],
    "unsupported_clause": ["포트를 8080으로 바꿔줘", "도커 컴포즈 파일도 만들어줘", "새 이미지 빌드해줘", "도커 허브에 푸시해줘", "환경변수 하나 추가해줘", "네트워크도 새로 만들어줘", "쉘 열어줘", "메모리 제한 걸어줘", "볼륨 백업해줘", "슬랙으로 알려줘"],
    "answers": {
        "slot_by_ref": ["{X}", "{X|으로} 해줘", "{X} 거", "{X|으로} 할게", "{X} 말한 거야", "{X} 그거", "{X|이요}"],
        "slot_by_index": ["{N}번", "{N}번으로 해줘", "{ORD} 거", "{ORD}", "{N}번이요"],
        "task_by_task": {
            "container.start": ["켜줘", "시작", "시작해줘", "켜는 거"],
            "container.stop": ["그냥 꺼줘", "중지", "멈추기만 해줘", "중지로 해줘"],
            "container.restart": ["재시작", "재시작해줘", "껐다 켜줘", "재시작으로"],
            "container.delete": ["삭제", "지워줘", "아예 삭제해줘", "삭제로 해줘"],
            "container.logs": ["로그부터 볼게", "로그 보여줘", "로그 확인", "로그만 볼래"]
        },
        "task_by_index": ["{N}번", "{ORD} 거", "{N}번으로 해줘"]
    },
    "corrections": {
        "change_target": ["{OLD} 말고 {NEW|으로} 해줘", "{NEW|으로} 바꿔줘", "대상을 {NEW|으로} 해줘", "아 {NEW|이었어}", "{OLD} 아니고 {NEW}"],
        "change_task": ["{OLDT} 말고 {NEWT|으로} 해줘", "{NEWT|으로} 바꿔줘", "아 {NEWT|으로} 할게", "{OLDT} 아니고 {NEWT}"],
        "remove": ["{X|은} 빼줘", "{X} 건 빼고 해줘", "{X|은} 하지 마", "{X|은} 제외해줘"],
        "cancel": ["그냥 취소할게", "취소", "됐어 하지 마", "다음에 할게", "아니 안 할래", "그만", "취소해줘"]
    },
    "ordinals": ["첫 번째", "두 번째", "세 번째", "네 번째", "다섯 번째", "여섯 번째", "일곱 번째", "여덟 번째"]
}
''')

# SLM 시스템 프롬프트 (학습·추론 공통)
SYSTEM_PROMPT = r"""
너는 Docker Desktop용 RPA 봇 생성기의 요청 해석기다. 사용자 발화를 읽고, 현재 계획을 갱신한 JSON만 출력한다.

[작업 목록]
container.start: 컨테이너 시작 (대상: container)
container.stop: 컨테이너 중지 (대상: container)
container.restart: 컨테이너 재시작 (대상: container)
container.delete: 컨테이너 삭제 (대상: container)
container.logs: 컨테이너 로그 보기 (대상: container)
image.run: 이미지로 새 컨테이너를 만들어 실행 (대상: image)
image.delete: 이미지 삭제 (대상: image)
image.prune: 사용하지 않는 이미지 정리 (대상 없음)
volume.delete: 볼륨 삭제 (대상: volume)

[출력 형식]
{"tasks":[{"task_candidates":["작업"],"args":{"대상종류":{"mention":"사용자 표현 또는 null","candidates":["이름"]}}}],"unsupported":["지원하지 않는 요청"],"cancel":false}

[규칙]
1. candidates에는 [환경]에 실제로 있는 이름만 넣는다. 사용자 표현에 해당하는 것을 모두 넣고, 해당하는 것이 없으면 빈 목록으로 둔다. 실행 상태나 사용 여부는 따지지 않는다.
2. 사용자가 대상을 말하지 않았으면 mention은 null, candidates는 빈 목록이다.
3. 작업이 애매하면 task_candidates에 가능한 작업을 모두 넣는다.
4. 작업 목록에 없는 요청은 tasks에 넣지 않고 unsupported에 원문 그대로 넣는다.
5. 요청이 여러 개면 말한 순서대로 tasks에 넣는다.
6. [현재 계획]이 있으면 이어서 고친 전체 계획을 출력한다. [직전 질문]의 선택지 번호로 답하면 그 선택지를 고른다.
7. 사용자가 취소하면 tasks는 빈 목록, cancel은 true로 한다.
8. 대상이 없는 작업은 args를 빈 객체로 둔다.
9. 컨테이너를 새로 만들거나 하나 더 띄우는 요청("만들고 실행", "새로 띄워")은 image.run 하나로, 이미 있는 컨테이너를 켜는 요청은 container.start로 처리한다.
""".strip()


# ============================================================================
# 2. 조사 처리
# ============================================================================
# 한국어 조사 붙이기. 영문 컨테이너 이름(nginx-proxy, shop-redis-1)도 읽는 소리 기준으로 받침을 판단한다.

# (받침 있음, ㄹ받침)
_DIGITS = {
    "0": (True, False), "1": (True, True), "2": (False, False), "3": (True, False),
    "4": (False, False), "5": (False, False), "6": (True, False), "7": (True, True),
    "8": (True, True), "9": (False, False),
}
# 단어 단위 발음 (사전에 없으면 마지막 알파벳 이름으로 판단)
_WORDS = {
    "nginx": (False, False), "redis": (False, False), "postgres": (False, False),
    "pg": (False, False), "mysql": (True, True), "mongo": (False, False),
    "rabbitmq": (False, False), "kafka": (False, False), "elasticsearch": (False, False),
    "grafana": (False, False), "prometheus": (False, False), "jenkins": (False, False),
    "minio": (False, False), "api": (False, False), "web": (True, False),
    "worker": (False, False), "admin": (True, False), "proxy": (False, False),
    "data": (False, False), "pgdata": (False, False), "esdata": (False, False),
    "home": (True, False), "latest": (False, False), "alpine": (True, False),
    "lts": (False, False), "management": (False, False), "test": (False, False),
    "dev": (False, False), "local": (True, True), "db": (False, False),
    "mq": (False, False), "ci": (False, False), "world": (True, True),
    "busybox": (False, False), "ubuntu": (False, False), "python": (True, False),
    "node": (False, False), "kafka-data": (False, False),
}
# 알파벳 이름: l(엘) m(엠) n(엔) r(알)만 받침이 있다
_LETTERS = {"l": (True, True), "m": (True, False), "n": (True, False), "r": (True, True)}


def batchim(text: str):
    """(받침 있음, ㄹ받침) 반환"""
    t = text.strip().rstrip("?!.~ ")
    if not t:
        return (False, False)
    ch = t[-1]
    if "가" <= ch <= "힣":
        jong = (ord(ch) - 0xAC00) % 28
        return (jong != 0, jong == 8)
    tokens = [x for x in re.split(r"[-_:./\s]+", t.lower()) if x]
    tok = tokens[-1] if tokens else t.lower()
    if tok in _WORDS:
        return _WORDS[tok]
    if tok[-1].isdigit():
        return _DIGITS[tok[-1]]
    if tok[-1].isalpha():
        return _LETTERS.get(tok[-1], (False, False))
    return (False, False)


def attach(word: str, p: str) -> str:
    """word에 조사 p를 붙인다. p는 '을'처럼 받침형 대표 표기로 쓴다."""
    if not p:
        return word
    b, rieul = batchim(word)
    table = {
        "을": "을" if b else "를",
        "이": "이" if b else "가",
        "은": "은" if b else "는",
        "과": "과" if b else "와",
        "랑": "이랑" if b else "랑",
        "으로": "으로" if (b and not rieul) else "로",
        "이요": "이요" if b else "요",
        "이었어": "이었어" if b else "였어",
        "의": "의",
        "도": "도",
        "하고": "하고",
    }
    if p not in table:
        raise ValueError(f"unknown josa: {p}")
    return word + table[p]


# ============================================================================
# 3. 대화 정책
# ============================================================================
# 대화 정책: SLM이 갱신한 계획(JSON)을 보고 '질문할지 / 확인 버튼을 띄울지 / 거절할지'를 결정한다.
# SLM은 학습 대상이고, 이 파일은 규칙 기반 코드라 학습하지 않는다. 데이터 생성과 실제 서비스에서 같은 코드를 쓴다.
#
# ctx 형식 (실서비스에서는 docker CLI로 채운다)
#   containers: [{"name", "image", "state": "running"|"exited"}]
#   images:     [{"name", "in_use": bool}]
#   volumes:    [{"name", "in_use": bool}]

SLOT_KO = {"container": "컨테이너", "image": "이미지", "volume": "볼륨"}
MAX_OPTIONS = 8


def _items(ctx, slot):
    return ctx[slot + "s"]


def _precond_ok(task_spec, slot, item):
    p = task_spec.get("precondition")
    if p is None:
        return True
    if p == "not_running":
        return item["state"] != "running"
    if p == "running":
        return item["state"] == "running"
    if p == "not_in_use":
        return not item["in_use"]
    raise ValueError(p)


def _fail_msg(task_spec, name):
    p = task_spec["precondition"]
    if p == "not_running":
        return f"{attach(name, '은')} 이미 실행 중이에요."
    if p == "running":
        return f"{attach(name, '은')} 이미 중지되어 있어요."
    return f"{attach(name, '은')} 사용 중이라 삭제할 수 없어요."


def _reason_short(task_spec):
    return {"not_running": "이미 실행 중이라", "running": "이미 중지되어 있어",
            "not_in_use": "사용 중이라"}[task_spec["precondition"]]


def _supported_summary(spec):
    labels = []
    for tid, t in spec.items():
        lab = t["label"]
        if t["slot"] == "container":
            lab = "컨테이너 " + lab
        labels.append(lab)
    return ", ".join(labels)


def decide(out, ctx, spec):
    """returns (bot_action: dict, next_state: dict | None)"""
    if out.get("cancel"):
        return {"type": "cancelled", "message": "요청을 취소했어요."}, None

    tasks = copy.deepcopy(out.get("tasks", []))
    unsupported = out.get("unsupported", [])
    notices = []
    if unsupported and tasks:
        notices.append("지원하지 않는 요청은 제외했어요: " + ", ".join(f"'{u}'" for u in unsupported))

    if not tasks:
        if unsupported:
            msg = "Docker Desktop에서 지원하지 않는 요청이에요. 가능한 작업: " + _supported_summary(spec)
        else:
            msg = "어떤 작업을 할지 이해하지 못했어요. 가능한 작업: " + _supported_summary(spec)
        return {"type": "reject", "message": msg}, None

    kept = []
    for i, t in enumerate(tasks):
        rest = tasks[i + 1:]
        cands_t = [c for c in t.get("task_candidates", []) if c in spec]
        if not cands_t:
            continue

        # 1) 작업이 애매함
        if len(cands_t) > 1:
            slot = spec[cands_t[0]]["slot"]
            mention = (t.get("args", {}).get(slot) or {}).get("mention") if slot else None
            msg = f"'{mention}'에 대해 어떤 작업을 할까요?" if mention else "어떤 작업을 할까요?"
            action = {"type": "ask", "kind": "task", "message": msg,
                      "options": [{"label": spec[c]["label"], "value": c} for c in cands_t],
                      "pending": {"task_index": len(kept), "field": "task"}, "notices": notices}
            return action, {"tasks": kept + [t] + rest}

        tid = cands_t[0]
        ts = spec[tid]
        slot = ts["slot"]
        if slot is None:
            t["args"] = {}
            kept.append(t)
            continue

        items = _items(ctx, slot)
        by_name = {it["name"]: it for it in items}
        arg = (t.get("args") or {}).get(slot) or {"mention": None, "candidates": []}
        mention = arg.get("mention")
        cands = [c for c in arg.get("candidates", []) if c in by_name]
        valid_all = [it["name"] for it in items if _precond_ok(ts, slot, it)]
        slot_ko = SLOT_KO[slot]

        def ask(kind, msg, options):
            action = {"type": "ask", "kind": kind, "message": msg,
                      "options": [{"label": o, "value": o} for o in options[:MAX_OPTIONS]],
                      "pending": {"task_index": len(kept), "field": slot}, "notices": notices}
            t["args"] = {slot: {"mention": mention, "candidates": cands}}
            return action, {"tasks": kept + [t] + rest}

        # 2) 대상을 말하지 않음
        if mention is None and not cands:
            if not valid_all:
                notices.append(ts["no_option"])
                continue
            return ask("missing", ts["question"], valid_all)

        # 3) 말했지만 환경에 없음
        if not cands:
            if not valid_all:
                notices.append(f"'{mention}'에 해당하는 {attach(slot_ko, '을')} 찾지 못했어요.")
                continue
            return ask("not_found", f"'{mention}'에 해당하는 {attach(slot_ko, '을')} 찾지 못했어요. 이 중에 있나요?", valid_all)

        # 4) 하나로 확정
        if len(cands) == 1:
            if _precond_ok(ts, slot, by_name[cands[0]]):
                t["args"] = {slot: {"mention": mention, "candidates": cands}}
                kept.append(t)
            else:
                notices.append(_fail_msg(ts, cands[0]))
            continue

        # 5) 여러 개 → 전제 조건으로 걸러본다
        valid = [c for c in cands if _precond_ok(ts, slot, by_name[c])]
        if not valid:
            notices.extend(_fail_msg(ts, c) for c in cands)
            continue
        if len(valid) == 1:
            excluded = [c for c in cands if c not in valid]
            notices.append(f"'{mention}' 중 {attach(', '.join(excluded), '은')} {_reason_short(ts)} "
                           f"{attach(valid[0], '을')} 골랐어요.")
            t["args"] = {slot: {"mention": mention, "candidates": valid}}
            kept.append(t)
            continue
        cands = valid
        return ask("ambiguous", f"'{mention}'에 해당하는 {attach(slot_ko, '이')} 여러 개예요. 어느 것인가요?", valid)

    # 중복 제거
    uniq, seen = [], set()
    for t in kept:
        key = (t["task_candidates"][0], tuple((t.get("args") or {}).get(spec[t["task_candidates"][0]]["slot"] or "", {}).get("candidates", [])))
        if key not in seen:
            seen.add(key)
            uniq.append(t)
    kept = uniq

    if not kept:
        return {"type": "reject", "message": " ".join(notices) or "실행할 작업이 없어요."}, None

    plan, destructive = [], False
    for t in kept:
        ts = spec[t["task_candidates"][0]]
        target = t["args"][ts["slot"]]["candidates"][0] if ts["slot"] else ""
        plan.append(ts["plan"].format(target=target))
        destructive = destructive or ts["destructive"]
    action = {"type": "confirm", "message": "아래 작업으로 RPA 봇을 생성할까요?", "plan": plan,
              "destructive": destructive,
              "warning": "되돌릴 수 없는 작업이 포함되어 있어요." if destructive else None,
              "buttons": ["예", "아니오"], "notices": notices}
    return action, {"tasks": kept}


CONFIRM_NO_QUESTION = {"type": "ask", "kind": "revise", "message": "어떤 부분을 바꿀까요?", "options": []}


# ============================================================================
# 4. 프롬프트 조립
# ============================================================================
# SLM 입력(프롬프트)과 출력 스키마. 학습 데이터와 실제 추론에서 똑같이 써야 한다.


def dumps(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def format_context(ctx):
    """실행 상태·사용 여부는 넣지 않는다 (판단은 정책 코드 몫)."""
    lines = ["[환경]", "컨테이너:"]
    lines += [f"- {c['name']} ({c['image']})" for c in ctx["containers"]] or ["- 없음"]
    lines.append("이미지:")
    lines += [f"- {i['name']}" for i in ctx["images"]] or ["- 없음"]
    lines.append("볼륨:")
    lines += [f"- {v['name']}" for v in ctx["volumes"]] or ["- 없음"]
    return "\n".join(lines)


def format_last_bot(action):
    if not action:
        return "없음"
    s = action["message"]
    opts = action.get("options") or []
    if opts:
        s += " " + " ".join(f"[{i + 1}] {o['label']}" for i, o in enumerate(opts))
    return s


def build_user_message(ctx, state, last_bot, user):
    return "\n".join([
        format_context(ctx),
        "[현재 계획]",
        dumps(state) if state else "없음",
        "[직전 질문]",
        format_last_bot(last_bot),
        "[사용자]",
        user,
    ])


def build_messages(ctx, state, last_bot, user, output=None):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(ctx, state, last_bot, user)}]
    if output is not None:
        msgs.append({"role": "assistant", "content": dumps(output)})
    return msgs


def build_schema(ctx, spec):
    """제약 디코딩용 JSON Schema. 요청마다 현재 환경 이름으로 enum을 채워서 없는 이름을 못 만들게 한다."""
    def arg_schema(names):
        cand = {"type": "array", "items": {"type": "string", "enum": names}} if names \
            else {"type": "array", "maxItems": 0}
        return {"type": "object", "additionalProperties": False, "required": ["mention", "candidates"],
                "properties": {"mention": {"type": ["string", "null"]}, "candidates": cand}}

    return {
        "type": "object", "additionalProperties": False,
        "required": ["tasks", "unsupported", "cancel"],
        "properties": {
            "tasks": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "required": ["task_candidates", "args"],
                "properties": {
                    "task_candidates": {"type": "array", "minItems": 1,
                                        "items": {"type": "string", "enum": list(spec)}},
                    "args": {"type": "object", "additionalProperties": False, "properties": {
                        "container": arg_schema([c["name"] for c in ctx["containers"]]),
                        "image": arg_schema([i["name"] for i in ctx["images"]]),
                        "volume": arg_schema([v["name"] for v in ctx["volumes"]]),
                    }},
                }}},
            "unsupported": {"type": "array", "items": {"type": "string"}},
            "cancel": {"type": "boolean"},
        },
    }


# ============================================================================
# 5. 데이터 생성기
# ============================================================================
# Docker Desktop RPA 봇 생성기 - SLM 학습 데이터 생성기
#
# 가상 Docker 환경(컨테이너·이미지·볼륨)을 무작위로 만들고, 시드 문장 패턴에 대상 표현을 끼워 넣어
# "환경 + 현재 계획 + 직전 질문 + 사용자 발화 → 갱신된 계획 JSON" 예시를 만든다.
# 정답 후보(candidates)는 환경 메타데이터로 계산하므로 라벨링이 필요 없다.
# 질문/확인 버튼 결정은 policy.decide()가 하고, 그 결과로 후속 턴(답변·수정·취소)을 이어서 만든다.

INFRA = LEX["infra_services"]
APP = LEX["app_services"]
SLOT_KEY = {"container": "C", "image": "I", "volume": "V"}
PH = re.compile(r"\{(\w+)(?:\|(\w+))?\}")

CASE_WEIGHTS = {
    "single_resolved": 24, "single_ambiguous": 9, "single_missing": 8, "single_not_found": 6,
    "task_ambiguous": 6, "prune": 3, "multi_task": 9, "multi_same": 4, "implicit_pair": 4,
    "unsupported": 7, "mixed": 4, "contrast_run_start": 7,
}
SINGLE_TASK_WEIGHTS = {
    "container.start": 12, "container.stop": 14, "container.restart": 12, "container.delete": 10,
    "container.logs": 12, "image.run": 11, "image.delete": 9, "volume.delete": 9,
}
RESOLVED_KIND_WEIGHTS = {"name": 35, "alias": 30, "category": 10, "project_alias": 20, "alias_tag": 5}
CONTAINER_TASKS = [t for t, s in SPEC.items() if s["slot"] == "container"]


def render(pattern, vals):
    def rep(m):
        text = vals[m.group(1)]
        return attach(text, m.group(2)) if m.group(2) else text
    return PH.sub(rep, pattern)


def svc_info(s):
    return INFRA.get(s) or APP.get(s) or {"aliases": [s], "categories": []}


def task_obj(tid, mention=None, cands=None):
    slot = SPEC[tid]["slot"]
    if slot is None:
        return {"task_candidates": [tid], "args": {}}
    return {"task_candidates": [tid], "args": {slot: {"mention": mention, "candidates": list(cands or [])}}}


def out_obj(tasks, unsupported=None, cancel=False):
    return {"tasks": tasks, "unsupported": unsupported or [], "cancel": cancel}


def target_of(t):
    slot = SPEC[t["task_candidates"][0]]["slot"]
    if not slot:
        return None, None
    c = t["args"][slot]["candidates"]
    return slot, (c[0] if len(c) == 1 else None)


class Gen:
    def __init__(self, split, seed):
        self.split = split
        self.r = random.Random(seed)
        self.unseen = False

    # ---------- 패턴 선택 (5번째마다 평가 전용) ----------
    def pick(self, lst, static=False):
        if len(lst) < 5:
            return self.r.choice(lst)
        held = [x for i, x in enumerate(lst) if i % 5 == 4]
        train = [x for i, x in enumerate(lst) if i % 5 != 4]
        if self.split == "train":
            return self.r.choice(train)
        if static or self.r.random() < 0.6:
            self.unseen = True
            return self.r.choice(held)
        return self.r.choice(train)

    def wchoice(self, weights):
        keys = list(weights)
        return self.r.choices(keys, weights=[weights[k] for k in keys])[0]

    # ---------- 가상 Docker 환경 ----------
    def make_context(self, force_dup=False):
        r, split = self.r, self.split
        n_proj = 2 if (force_dup or r.random() < 0.4) else 1
        projs = r.sample(LEX["projects"][split], n_proj)
        style = r.choice(["compose", "compose", "underscore"] + (["plain"] if n_proj == 1 else []))
        shared = r.choice(list(INFRA)) if (force_dup and n_proj == 2) else None
        containers, names, tags = [], set(), {}

        def cname(p, s):
            if style == "compose":
                return f"{p}-{s}-1"
            if style == "underscore":
                return f"{p}_{s}"
            return s if s in INFRA else f"{p}-{s}"

        def add(name, s, p):
            if name in names:
                return
            names.add(name)
            if s in INFRA:
                tag = tags.setdefault((s, p), r.choice(INFRA[s]["tags"]))
                image = f"{INFRA[s]['repo']}:{tag}"
            else:
                image = f"{p}-{s}:{r.choice(APP[s]['tags'])}"
            containers.append({"name": name, "image": image,
                               "state": "running" if r.random() < 0.65 else "exited",
                               "service": s, "project": p})

        for p in projs:
            svcs = r.sample(list(APP), r.randint(0, 2)) + r.sample(list(INFRA), r.randint(1, 3))
            if shared and shared not in svcs:
                svcs.append(shared)
            r.shuffle(svcs)
            for s in svcs:
                add(cname(p, s), s, p)
        for _ in range(r.choice([0, 0, 1, 2])):
            s = r.choice(list(INFRA))
            add(r.choice(LEX["standalone_name_forms"][split]).format(s=s), s, None)

        images, seen = [], set()
        for c in containers:
            if c["image"] not in seen:
                seen.add(c["image"])
                images.append({"name": c["image"], "in_use": True, "service": c["service"],
                               "project": c["project"] if c["service"] in APP else None})
        for _ in range(r.randint(0, 3)):
            infra_present = [c["service"] for c in containers if c["service"] in INFRA]
            if r.random() < 0.6 and infra_present:
                s = r.choice(infra_present)
                name, svc = f"{INFRA[s]['repo']}:{r.choice(INFRA[s]['tags'])}", s
            else:
                name = r.choice(LEX["extra_images"])
                svc = name.split(":")[0]
            if name not in seen:
                seen.add(name)
                images.append({"name": name, "in_use": False, "service": svc, "project": None})

        volumes, vseen = [], set()
        for c in containers:
            s = c["service"]
            if s not in INFRA or not INFRA[s]["volume"]:
                continue
            vol = INFRA[s]["volume"]
            if c["project"] and style != "plain":
                name = f"{c['project']}_{vol}"
            else:
                name = vol if vol not in vseen else f"{c['name']}-data"
            if name in vseen:
                continue
            vseen.add(name)
            volumes.append({"name": name, "in_use": True, "service": s,
                            "project": c["project"] if (c["project"] and c["project"] in name) else None})
        if r.random() < 0.4:
            s = r.choice([k for k in INFRA if INFRA[k]["volume"]])
            name = f"{r.choice(LEX['orphan_volume_prefixes'][split])}_{INFRA[s]['volume']}"
            if name not in vseen:
                volumes.append({"name": name, "in_use": False, "service": s, "project": None})
        return {"containers": containers, "images": images, "volumes": volumes}

    # ---------- 대상 표현 ----------
    @staticmethod
    def enumerate_refs(ctx, slot):
        """(종류, 표현, 정답 후보 목록)을 모두 나열한다.
        같은 표현이 여러 서비스에 걸리면 후보를 합친다. 예: '웹' = web 앱(별칭) + nginx(역할 이름)"""
        items = ctx[slot + "s"]
        order = {it["name"]: i for i, it in enumerate(items)}
        refs = [("name", it["name"], [it["name"]]) for it in items]
        table = {}  # 표현 -> {"kinds": set, "names": set}

        def add(text, kind, name):
            e = table.setdefault(text, {"kinds": set(), "names": set()})
            e["kinds"].add(kind)
            e["names"].add(name)

        for it in items:
            info = svc_info(it["service"])
            words = info["aliases"] + info.get("categories", [])
            for a in info["aliases"]:
                add(a, "alias", it["name"])
            for c in info.get("categories", []):
                add(c, "category", it["name"])
            p = it.get("project")
            if p and p in it["name"]:
                for w in words:
                    for form in ("{p} {a}", "{p} 쪽 {a}", "{p}의 {a}"):
                        add(form.format(p=p, a=w), "project_alias", it["name"])
            if slot == "image" and it["service"] in INFRA:
                tag = it["name"].split(":", 1)[1]
                for a in info["aliases"]:
                    add(f"{a} {tag}", "alias_tag", it["name"])
        for text, e in table.items():
            kind = "alias" if "alias" in e["kinds"] else sorted(e["kinds"])[0]
            refs.append((kind, text, sorted(e["names"], key=order.get)))
        missing = set()
        for s in list(INFRA) + list(APP):
            if slot == "volume" and not (s in INFRA and INFRA[s]["volume"]):
                continue
            info = svc_info(s)
            for w in info["aliases"] + info.get("categories", []):
                if w not in table:
                    missing.add(w)
        refs += [("not_found", w, []) for w in sorted(missing)]
        # 표현이 어떤 항목의 이름과 똑같으면 그 이름 하나를 가리키는 것으로 정한다 (라벨 일관성)
        names = set(order)
        return [x for x in refs if x[0] == "name" or x[1] not in names]

    def pick_ref(self, ctx, slot, bucket):
        refs = self.enumerate_refs(ctx, slot)
        if bucket == "resolved":
            by_kind = {}
            for ref in refs:
                if len(ref[2]) == 1 and ref[0] != "not_found":
                    by_kind.setdefault(ref[0], []).append(ref)
            if not by_kind:
                return None
            kind = self.wchoice({k: w for k, w in RESOLVED_KIND_WEIGHTS.items() if k in by_kind})
            return self.r.choice(by_kind[kind])
        if bucket == "ambiguous":
            pool = [x for x in refs if len(x[2]) >= 2]
        else:
            pool = [x for x in refs if x[0] == "not_found"]
        return self.r.choice(pool) if pool else None

    def unique_ref(self, ctx, slot, name):
        """name 하나만 가리키는 표현 (절반은 이름 그대로)"""
        pool = [x for x in self.enumerate_refs(ctx, slot) if x[2] == [name] and x[0] != "name"]
        if pool and self.r.random() < 0.5:
            return self.r.choice(pool)
        return ("name", name, [name])

    def ctx_with_ref(self, slot, bucket, tries=30):
        for i in range(tries):
            ctx = self.make_context(force_dup=(bucket == "ambiguous" and i % 2 == 1))
            ref = self.pick_ref(ctx, slot, bucket)
            if ref:
                return ctx, ref
        return None, None

    # ---------- 첫 턴 케이스 ----------
    def case_single(self, bucket):
        tid = self.wchoice(SINGLE_TASK_WEIGHTS)
        slot = SPEC[tid]["slot"]
        seeds = SEEDS["tasks"][tid]
        if bucket == "missing":
            return self.make_context(), self.pick(seeds["final_noslot"], static=True), out_obj([task_obj(tid, None, [])])
        ctx, ref = self.ctx_with_ref(slot, bucket)
        if not ctx:
            return None
        user = render(self.pick(seeds["final"]), {SLOT_KEY[slot]: ref[1]})
        return ctx, user, out_obj([task_obj(tid, ref[1], ref[2])])

    def case_single_resolved(self):
        return self.case_single("resolved")

    def case_single_ambiguous(self):
        return self.case_single("ambiguous")

    def case_single_missing(self):
        return self.case_single("missing")

    def case_single_not_found(self):
        return self.case_single("not_found")

    def case_task_ambiguous(self):
        entry = self.pick(SEEDS["ambiguous_task"])
        ctx, ref = self.ctx_with_ref("container", "resolved" if self.r.random() < 0.8 else "ambiguous")
        if not ctx:
            return None
        user = render(entry["pattern"], {"C": ref[1]})
        t = {"task_candidates": list(entry["candidates"]),
             "args": {"container": {"mention": ref[1], "candidates": ref[2]}}}
        return ctx, user, out_obj([t])

    def case_prune(self):
        return self.make_context(), self.pick(SEEDS["tasks"]["image.prune"]["final_noslot"], static=True), \
            out_obj([task_obj("image.prune")])

    def _part(self, ctx, tid, form, avoid=None):
        """복합 요청의 한 조각: (문장 조각, task 객체)"""
        seeds = SEEDS["tasks"][tid]
        slot = SPEC[tid]["slot"]
        if slot is None:
            lst = seeds[form] or seeds["final_noslot"]
            return self.pick(lst, static=True), task_obj(tid)
        bucket = self.wchoice({"resolved": 80, "ambiguous": 10, "not_found": 10})
        ref = None
        for _ in range(5):
            ref = self.pick_ref(ctx, slot, bucket) or self.pick_ref(ctx, slot, "resolved")
            if ref and ref[2] != avoid:
                break
        if not ref:
            return None, None
        return render(self.pick(seeds[form]), {SLOT_KEY[slot]: ref[1]}), task_obj(tid, ref[1], ref[2])

    def case_contrast_run_start(self):
        """같은 표현으로 '있는 컨테이너 켜기'와 '새 컨테이너 만들기'를 구분하게 하는 대조 예시"""
        for _ in range(30):
            ctx = self.make_context()
            c_refs = {x[1]: x for x in self.enumerate_refs(ctx, "container") if x[0] not in ("name", "not_found")}
            i_refs = {x[1]: x for x in self.enumerate_refs(ctx, "image") if x[0] not in ("name", "not_found")}
            common = sorted(set(c_refs) & set(i_refs))
            if common:
                break
        else:
            return None
        # '웹'은 앱 web과 nginx가 겹치는 대표 표현이라 비중을 높인다
        text = "웹" if ("웹" in common and self.r.random() < 0.4) else self.r.choice(common)
        if self.r.random() < 0.5:
            pats = [p for p in SEEDS["tasks"]["image.run"]["final"] if "이미지" not in p and not p.startswith("run")]
            ref, tid, key = i_refs[text], "image.run", "I"
        else:
            pats = SEEDS["tasks"]["container.start"]["final"]
            ref, tid, key = c_refs[text], "container.start", "C"
        user = render(self.pick(pats), {key: text})
        return ctx, user, out_obj([task_obj(tid, ref[1], ref[2])])

    def case_multi_task(self):
        t1, t2 = self.r.sample(list(SPEC), 2)
        if {t1, t2} == {"image.run", "container.start"}:
            return None  # "만들고 실행"이 작업 2개라는 잘못된 신호를 주지 않도록 뺀다
        ctx = self.make_context()
        p1, o1 = self._part(ctx, t1, "conn")
        if p1 is None:
            return None
        avoid = o1["args"].get(SPEC[t2]["slot"], {}).get("candidates") if SPEC[t2]["slot"] else None
        p2, o2 = self._part(ctx, t2, "final", avoid=avoid)
        if p2 is None:
            return None
        return ctx, f"{p1} {p2}", out_obj([o1, o2])

    def case_multi_same(self):
        tid = self.wchoice({**{t: 15 for t in CONTAINER_TASKS}, "image.delete": 8, "volume.delete": 7})
        slot = SPEC[tid]["slot"]
        for _ in range(20):
            ctx = self.make_context()
            a = self.pick_ref(ctx, slot, "resolved")
            b = self.pick_ref(ctx, slot, "resolved")
            if a and b and a[2] != b[2]:
                break
        else:
            return None
        comp = render(self.pick(SEEDS["multi_same_joiners"]), {"A": a[1], "B": b[1]})
        pats = [p for p in SEEDS["tasks"][tid]["final"] if not re.match(r"[a-z]", p)]
        user = render(self.pick(pats), {SLOT_KEY[slot]: comp})
        return ctx, user, out_obj([task_obj(tid, a[1], a[2]), task_obj(tid, b[1], b[2])])

    def case_implicit_pair(self):
        t1, t2 = self.r.choice(SEEDS["implicit_pairs"])
        ctx, ref = self.ctx_with_ref("container", "resolved" if self.r.random() < 0.85 else "ambiguous")
        if not ctx:
            return None
        user = render(self.pick(SEEDS["tasks"][t1]["conn"]), {"C": ref[1]}) + " " + \
            self.pick(SEEDS["tasks"][t2]["implicit"])
        return ctx, user, out_obj([task_obj(t1, ref[1], ref[2]), task_obj(t2, ref[1], ref[2])])

    def case_unsupported(self):
        pat = self.pick(SEEDS["unsupported"], static=True)
        ctx = self.make_context()
        vals = {}
        if "{C" in pat:
            ref = self.pick_ref(ctx, "container", "resolved")
            if not ref:
                return None
            vals["C"] = ref[1]
        user = render(pat, vals)
        return ctx, user, out_obj([], unsupported=[user])

    def case_mixed(self):
        tid = self.r.choice([t for t in SPEC if SEEDS["tasks"][t]["conn"]])
        ctx = self.make_context()
        part, obj = self._part(ctx, tid, "conn")
        if part is None:
            return None
        clause = self.pick(SEEDS["unsupported_clause"])
        return ctx, f"{part} {clause}", out_obj([obj], unsupported=[clause])

    # ---------- 후속 턴 ----------
    def answer_ask(self, ctx, action, state):
        r = self.r
        if r.random() < 0.06:
            return self.pick(SEEDS["corrections"]["cancel"]), out_obj([], cancel=True), "cancel"
        pend = action["pending"]
        k, field = pend["task_index"], pend["field"]
        tasks = copy.deepcopy(state["tasks"])
        opts = action["options"]
        j = r.randrange(len(opts))
        chosen = opts[j]["value"]
        nv = {"N": str(j + 1), "ORD": SEEDS["ordinals"][j]}
        ans = SEEDS["answers"]
        if field == "task":
            if r.random() < 0.6:
                user = self.pick(ans["task_by_task"][chosen])
                how = "answer_task"
            else:
                user = render(self.pick(ans["task_by_index"]), nv)
                how = "answer_task_index"
            tasks[k]["task_candidates"] = [chosen]
        else:
            if r.random() < 0.6:
                ref = self.unique_ref(ctx, field, chosen)
                user = render(self.pick(ans["slot_by_ref"]), {"X": ref[1]})
                mention, how = ref[1], "answer_slot"
            else:
                pat = self.pick(ans["slot_by_index"])
                user = render(pat, nv)
                mention = f"{j + 1}번" if "{N}" in pat else SEEDS["ordinals"][j]
                how = "answer_slot_index"
            # "웹 로그 보여주고 재시작해줘"처럼 같은 대상을 공유하는 작업에는 답을 모두 적용한다
            asked = tasks[k]["args"].get(field)
            for t in tasks:
                if t["args"].get(field) == asked:
                    t["args"] = {field: {"mention": mention, "candidates": [chosen]}}
        return user, out_obj(tasks), how

    def revise(self, ctx, state):
        """확인 단계에서 '아니오'를 누른 뒤의 수정 발화"""
        r, cor = self.r, SEEDS["corrections"]
        tasks = state["tasks"]
        targets = [target_of(t) for t in tasks]
        distinct = {tg for tg in targets if tg[1]}
        ops = {"change_target": 30, "change_task": 20, "add": 25, "remove": 10, "cancel": 15}
        for _ in range(10):
            op = self.wchoice(ops)
            if op == "cancel":
                return self.pick(cor["cancel"]), out_obj([], cancel=True), "revise_cancel"

            if op == "change_target" and distinct:
                slot, cur = r.choice(sorted(distinct))
                others = [it["name"] for it in ctx[slot + "s"] if it["name"] != cur]
                if not others:
                    continue
                new = r.choice(others)
                newref, oldref = self.unique_ref(ctx, slot, new), self.unique_ref(ctx, slot, cur)
                pats = cor["change_target"] if len(distinct) == 1 else [p for p in cor["change_target"] if "{OLD}" in p]
                user = render(self.pick(pats), {"OLD": oldref[1], "NEW": newref[1]})
                out = copy.deepcopy(tasks)
                for t in out:
                    if target_of(t) == (slot, cur):
                        t["args"][slot] = {"mention": newref[1], "candidates": [new]}
                return user, out_obj(out), "revise_target"

            if op == "change_task":
                tids = [t["task_candidates"][0] for t in tasks]
                idx = [i for i, tid in enumerate(tids) if tid in CONTAINER_TASKS and tids.count(tid) == 1]
                if not idx:
                    continue
                k = r.choice(idx)
                old = tids[k]
                new = r.choice([t for t in CONTAINER_TASKS if t != old and t not in tids])
                pats = cor["change_task"] if len(tasks) == 1 else [p for p in cor["change_task"] if "{OLDT}" in p]
                user = render(self.pick(pats), {"OLDT": SPEC[old]["label"], "NEWT": SPEC[new]["label"]})
                out = copy.deepcopy(tasks)
                out[k]["task_candidates"] = [new]
                return user, out_obj(out), "revise_task"

            if op == "add":
                tid = self.wchoice({**{t: 12 for t in CONTAINER_TASKS}, "image.delete": 5,
                                    "image.run": 4, "volume.delete": 5, "image.prune": 4})
                slot = SPEC[tid]["slot"]
                cont_targets = {tg[1] for tg in distinct if tg[0] == "container"}
                if slot is None:
                    if any(t["task_candidates"][0] == tid for t in tasks):
                        continue
                    new_t = task_obj(tid)
                    user = self.pick(SEEDS["tasks"][tid]["add"])
                elif tid == "container.logs" and len(cont_targets) == 1 and r.random() < 0.4:
                    name = next(iter(cont_targets))
                    user = render(self.pick(SEEDS["tasks"][tid]["add"]), {"C": "그거"})
                    new_t = task_obj(tid, "그거", [name])
                else:
                    ref = self.pick_ref(ctx, slot, "resolved" if r.random() < 0.85 else "ambiguous")
                    if not ref or any((slot, n) in distinct for n in ref[2]):
                        continue  # 이미 계획에 있는 대상에 다른 작업을 붙이면 '중지+시작'처럼 모순이 생긴다
                    user = render(self.pick(SEEDS["tasks"][tid]["add"]), {SLOT_KEY[slot]: ref[1]})
                    new_t = task_obj(tid, ref[1], ref[2])
                if any(t["task_candidates"] == new_t["task_candidates"] and t["args"] == new_t["args"] for t in tasks):
                    continue
                if any(t["task_candidates"] == new_t["task_candidates"] and target_of(t) == target_of(new_t)
                       and target_of(t)[1] for t in tasks):
                    continue
                return user, out_obj(copy.deepcopy(tasks) + [new_t]), "revise_add"

            if op == "remove" and len(tasks) >= 2:
                once = [tg for tg in distinct if targets.count(tg) == 1]
                if not once:
                    continue
                slot, name = r.choice(sorted(once))
                ref = self.unique_ref(ctx, slot, name)
                user = render(self.pick(cor["remove"]), {"X": ref[1]})
                out = [copy.deepcopy(t) for t in tasks if target_of(t) != (slot, name)]
                return user, out_obj(out), "revise_remove"
        return None

    def follow(self, base_case, ctx, action, state, turn, batch):
        if turn > 4 or action is None:
            return
        self.unseen = False
        if action["type"] == "ask":
            if self.r.random() > 0.75:
                return
            user, output, how = self.answer_ask(ctx, action, state)
            last_bot = action
        elif action["type"] == "confirm":
            if self.r.random() > 0.22:
                return
            res = self.revise(ctx, state)
            if not res:
                return
            user, output, how = res
            last_bot = CONFIRM_NO_QUESTION
        else:
            return
        rec, a2, s2 = self.record(f"followup:{how}", ctx, state, last_bot, user, output, turn, base_case)
        batch.append(rec)
        self.follow(base_case, ctx, a2, s2, turn + 1, batch)

    def record(self, case, ctx, state, last_bot, user, output, turn, base_case=None):
        action, next_state = decide(output, ctx, SPEC)
        rec = {"split": self.split, "case": case, "base_case": base_case or case, "turn": turn,
               "unseen_pattern": self.unseen, "context": ctx, "state": state,
               "last_bot": {k: last_bot[k] for k in ("type", "kind", "message", "options") if k in last_bot} if last_bot else None,
               "user": user, "output": output, "bot_action": action}
        return rec, action, next_state

    def generate(self, n):
        recs, seen = [], set()
        while len(recs) < n:
            case = self.wchoice(CASE_WEIGHTS)
            self.unseen = False
            res = getattr(self, "case_" + case)()
            if res is None:
                continue
            ctx, user, output = res
            rec, action, state = self.record(case, ctx, None, None, user, output, 1)
            batch = [rec]
            self.follow(case, ctx, action, state, 2, batch)
            for b in batch:
                key = (format_context(b["context"]), dumps(b["state"]), format_last_bot(b["last_bot"]), b["user"])
                if key in seen:
                    continue
                seen.add(key)
                recs.append(b)
        return recs[:n]


def cmd_gen(n_train=1200, n_eval=250, seed=42, out="data", validate=True):
    """train/eval 데이터를 out 폴더에 만든다. 반환값: 검증 오류 수"""
    os.makedirs(out, exist_ok=True)
    stats = {}
    for split, n, sd in (("train", n_train, seed), ("eval", n_eval, seed + 1000)):
        recs = Gen(split, sd).generate(n)
        for i, rec in enumerate(recs):
            rec["id"] = f"{split}-{i:05d}"
        with open(os.path.join(out, f"{split}.jsonl"), "w", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps({"id": rec["id"], **{k: v for k, v in rec.items() if k != "id"}},
                                   ensure_ascii=False) + "\n")
        with open(os.path.join(out, f"{split}_sft.jsonl"), "w", encoding="utf-8") as f:
            for rec in recs:
                msgs = build_messages(rec["context"], rec["state"], rec["last_bot"], rec["user"], rec["output"])
                f.write(json.dumps({"id": rec["id"], "messages": msgs}, ensure_ascii=False) + "\n")
        stats[split] = {
            "examples": len(recs),
            "by_base_case": dict(Counter(r["base_case"] for r in recs).most_common()),
            "by_case": dict(Counter(r["case"] for r in recs).most_common()),
            "by_turn": dict(sorted(Counter(r["turn"] for r in recs).items())),
            "by_bot_action": dict(Counter(r["bot_action"]["type"] + (":" + r["bot_action"]["kind"] if r["bot_action"].get("kind") else "")
                                          for r in recs).most_common()),
            "unseen_pattern": sum(r["unseen_pattern"] for r in recs),
        }
    with open(os.path.join(out, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    for split, s in stats.items():
        print(f"[{split}] {s['examples']}개 생성 → {out}/{split}.jsonl, {out}/{split}_sft.jsonl")
    print(f"케이스별 개수는 {out}/stats.json 참고\n")
    return cmd_validate(out) if validate else 0


# ============================================================================
# 6. 검증·채점·미리보기
# ============================================================================
# 생성된 데이터 검증: 스키마, 후보 정합성(독립 재계산), 선택지 번호 답변, 조사·자리표시자 오류, train/eval 중복.

def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def aliases(s):
    return (INFRA.get(s) or APP.get(s) or {"aliases": [s]})["aliases"]


def categories(s):
    return (INFRA.get(s) or APP.get(s) or {}).get("categories", [])


def resolve(mention, ctx, slot):
    """생성기와 별개로 작성한 해석기: 표현 → 후보 이름 목록"""
    items = ctx[slot + "s"]
    hit = [it["name"] for it in items if it["name"] == mention]
    if hit:
        return hit
    out = []
    for it in items:
        s, p = it["service"], it.get("project")
        al = aliases(s)
        words = al + categories(s)
        ok = mention in words
        if p and p in it["name"]:
            ok = ok or any(mention in (f"{p} {a}", f"{p} 쪽 {a}", f"{p}의 {a}") for a in words)
        if slot == "image" and s in INFRA and ":" in it["name"]:
            ok = ok or any(mention == f"{a} {it['name'].split(':', 1)[1]}" for a in al)
        if ok:
            out.append(it["name"])
    return out


def check(recs, split):
    errs = []
    ords = SEEDS["ordinals"]
    for r in recs:
        rid, ctx, out = r["id"], r["context"], r["output"]
        names = {s: {it["name"] for it in ctx[s + "s"]} for s in ("container", "image", "volume")}
        if set(out) != {"tasks", "unsupported", "cancel"}:
            errs.append((rid, "top-level keys"))
        if out["cancel"] and out["tasks"]:
            errs.append((rid, "cancel with tasks"))
        carried = [json.dumps(x, sort_keys=True) for x in ((r["state"] or {}).get("tasks") or [])]
        for t in out["tasks"]:
            if json.dumps(t, sort_keys=True) in carried:
                continue  # 현재 계획에서 그대로 이어받은 작업
            tc = t["task_candidates"]
            if not tc or any(x not in SPEC for x in tc):
                errs.append((rid, f"bad task {tc}"))
                continue
            slots = {SPEC[x]["slot"] for x in tc}
            if len(slots) != 1:
                errs.append((rid, f"mixed slots {tc}"))
            slot = slots.pop()
            if slot is None:
                if t["args"] != {}:
                    errs.append((rid, "slotless args not empty"))
                continue
            if set(t["args"]) != {slot}:
                errs.append((rid, f"arg key {list(t['args'])} != {slot}"))
                continue
            a = t["args"][slot]
            if set(a) != {"mention", "candidates"}:
                errs.append((rid, "arg keys"))
            if any(c not in names[slot] for c in a["candidates"]):
                errs.append((rid, f"candidate not in ctx {a['candidates']}"))
            m = a["mention"]
            prev_args = [json.dumps(a2, sort_keys=True) for t2 in ((r["state"] or {}).get("tasks") or [])
                         for a2 in t2.get("args", {}).values()]
            if json.dumps(a, sort_keys=True) in prev_args:
                continue  # 대상은 이전 계획 그대로
            if m is None:
                if a["candidates"]:
                    errs.append((rid, "null mention with candidates"))
                continue
            # 선택지 번호 답변 / 지시어는 직전 질문·계획으로 확인
            idx = None
            mm = re.fullmatch(r"(\d)번", m)
            if mm:
                idx = int(mm.group(1)) - 1
            elif m in ords:
                idx = ords.index(m)
            if idx is not None:
                opts = (r["last_bot"] or {}).get("options") or []
                if idx >= len(opts) or a["candidates"] != [opts[idx]["value"]]:
                    errs.append((rid, f"index answer mismatch {m}"))
                continue
            if m == "그거":
                continue
            exp = resolve(m, ctx, slot)
            if exp != a["candidates"]:
                errs.append((rid, f"resolve mismatch '{m}': gold={a['candidates']} indep={exp}"))
        # 텍스트 이상
        if re.search(r"[{}]|None|을를|를을|이가|은는|으로로", r["user"]):
            errs.append((rid, f"text artifact: {r['user']}"))
        for u in out["unsupported"]:
            if u not in r["user"]:
                errs.append((rid, "unsupported span not in user text"))
    return errs


def check_sft(path, recs):
    errs = []
    by_id = {r["id"]: r for r in recs}
    for line in load(path):
        msgs = line["messages"]
        if [m["role"] for m in msgs] != ["system", "user", "assistant"]:
            errs.append((line["id"], "roles"))
        if json.loads(msgs[2]["content"]) != by_id[line["id"]]["output"]:
            errs.append((line["id"], "assistant != output"))
    return errs


def cmd_validate(d="data"):
    """반환값: 오류 수"""
    tr, ev = load(os.path.join(d, "train.jsonl")), load(os.path.join(d, "eval.jsonl"))
    total = 0
    for split, recs in (("train", tr), ("eval", ev)):
        errs = check(recs, split) + check_sft(os.path.join(d, f"{split}_sft.jsonl"), recs)
        total += len(errs)
        print(f"[{split}] {len(recs)} examples, {len(errs)} errors")
        for e in errs[:20]:
            print("   ", e)
    # train/eval 중복
    tr_keys = {(dumps(r["context"]), r["user"]) for r in tr}
    dup = sum((dumps(r["context"]), r["user"]) in tr_keys for r in ev)
    tr_users = {r["user"] for r in tr}
    same_user = [r["user"] for r in ev if r["user"] in tr_users]
    print(f"[leak] eval의 (환경, 발화) 완전 중복: {dup}건 / 발화 문자열만 같은 것: {len(same_user)}건 (예: {same_user[:5]})")
    proj_tr = set(LEX["projects"]["train"])
    leak_proj = sum(any(p in r["user"] for p in proj_tr) for r in ev)
    print(f"[leak] eval 발화에 train 전용 프로젝트명 등장: {leak_proj}건")
    lens = sorted(len(json.dumps(json.loads(line)["messages"], ensure_ascii=False))
                  for line in open(os.path.join(d, "train_sft.jsonl"), encoding="utf-8"))
    print(f"[len] 예시 길이(문자) 중앙값 {lens[len(lens) // 2]}, 최대 {lens[-1]}")
    return total

# 모델 예측 채점.
# 예측 파일 형식(JSONL): {"id": "eval-00001", "prediction": "<모델이 출력한 원문 문자열>"}
#
# 지표
# - json_valid   : 출력이 JSON으로 파싱되는 비율
# - exact        : 정답 JSON과 완전히 같은 비율
# - tasks/args   : 작업 목록 / 대상(mention 제외, candidates만) 일치율
# - behavior     : 예측을 정책에 넣었을 때 챗봇 행동(질문 종류·선택지·실행 계획)이 정답과 같은 비율  ← 실사용 품질에 가장 가까운 지표

def parse(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").split("\n", 1)[-1]
    try:
        obj = json.loads(text[text.index("{"): text.rindex("}") + 1])
        return obj if isinstance(obj, dict) else None
    except (ValueError, json.JSONDecodeError):
        return None


def behavior(action):
    if action is None:
        return None
    return (action["type"], action.get("kind"),
            tuple(o["value"] for o in action.get("options") or []), tuple(action.get("plan") or []))


def cmd_score(gold_path, pred_path):
    """pred_path: {"id", "prediction"} JSONL. 틀린 예시는 <pred_path>_fails.jsonl로 저장"""
    gold = {json.loads(line)["id"]: json.loads(line) for line in open(gold_path, encoding="utf-8")}
    preds = {json.loads(line)["id"]: json.loads(line)["prediction"] for line in open(pred_path, encoding="utf-8")}
    m = defaultdict(lambda: defaultdict(int))
    fails = []
    for gid, g in gold.items():
        if gid not in preds:
            continue
        p = parse(preds[gid])
        groups = ["all", "case:" + g["base_case"], "turn1" if g["turn"] == 1 else "followup",
                  "unseen_pattern" if g["unseen_pattern"] else "seen_pattern"]
        res = {"n": 1, "json_valid": p is not None}
        if p is not None:
            go = g["output"]
            res["exact"] = p == go
            res["tasks"] = [t.get("task_candidates") for t in p.get("tasks", [])] == [t["task_candidates"] for t in go["tasks"]]
            strip = lambda o: [{k: v.get("candidates") for k, v in (t.get("args") or {}).items()} for t in o.get("tasks", [])]
            res["args"] = strip(p) == strip(go)
            try:
                pa, _ = decide(p, g["context"], SPEC)
            except Exception:  # 형식이 깨진 출력
                pa = None
            res["behavior"] = behavior(pa) == behavior(g["bot_action"])
        if not res.get("behavior"):
            fails.append({"id": gid, "user": g["user"], "gold": g["output"], "pred": preds[gid]})
        for grp in groups:
            for k, v in res.items():
                m[grp][k] += int(bool(v))

    keys = ["json_valid", "exact", "tasks", "args", "behavior"]
    print(f"{'group':<28}{'n':>5}" + "".join(f"{k:>12}" for k in keys))
    for grp in sorted(m, key=lambda x: (x != "all", x)):
        n = m[grp]["n"]
        print(f"{grp:<28}{n:>5}" + "".join(f"{m[grp][k] / n:>12.1%}" for k in keys))
    out = os.path.splitext(pred_path)[0] + "_fails.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for x in fails:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"\n틀린 예시 {len(fails)}건 → {out}")


def cmd_preview(path="data/train.jsonl", n=30, seed=0):
    """사람이 눈으로 검수하기 좋게 무작위 예시를 마크다운으로 출력한다 (> preview.md 로 저장)"""
    recs = [json.loads(line) for line in open(path, encoding="utf-8")]
    random.Random(seed).shuffle(recs)

    print(f"# 데이터 미리보기 ({path}, {n}개 무작위)\n")
    for r in recs[:n]:
        ctx = r["context"]
        print(f"## {r['id']} · {r['case']} · turn {r['turn']}")
        print("- 컨테이너: " + ", ".join(f"{c['name']}({'실행' if c['state'] == 'running' else '중지'})" for c in ctx["containers"]))
        print("- 이미지: " + ", ".join(i["name"] for i in ctx["images"]))
        print("- 볼륨: " + (", ".join(v["name"] for v in ctx["volumes"]) or "없음"))
        if r["state"]:
            print(f"- 현재 계획: `{json.dumps(r['state'], ensure_ascii=False)}`")
        if r["last_bot"]:
            lb = r["last_bot"]
            opts = " ".join(f"[{i + 1}] {o['label']}" for i, o in enumerate(lb.get("options") or []))
            print(f"- 직전 질문: {lb['message']} {opts}")
        print(f"- **사용자**: {r['user']}")
        print(f"- **SLM 정답**: `{json.dumps(r['output'], ensure_ascii=False)}`")
        a = r["bot_action"]
        line = f"- **챗봇 다음 행동**: {a['type']}"
        if a["type"] == "ask":
            line += f" — {a['message']} " + " ".join(f"[{o['label']}]" for o in a["options"])
        elif a["type"] == "confirm":
            line += " — " + " / ".join(a["plan"]) + " → [예] [아니오]"
            if a.get("warning"):
                line += f" ⚠ {a['warning']}"
        else:
            line += f" — {a['message']}"
        print(line)
        if a.get("notices"):
            print("- 안내: " + " ".join(a["notices"]))
        print()


# ============================================================================
# 7. 명령줄 진입점
# ============================================================================

def main(argv=None):
    ap = argparse.ArgumentParser(description="Docker Desktop RPA 봇 생성기 - SLM 학습 데이터 도구")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen", help="학습/평가 데이터 생성 (끝나면 자동 검증)")
    g.add_argument("--train", type=int, default=1200)
    g.add_argument("--eval", type=int, default=250)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--out", default="data")
    v = sub.add_parser("validate", help="데이터 검증")
    v.add_argument("dir", nargs="?", default="data")
    s = sub.add_parser("score", help="모델 예측 채점")
    s.add_argument("gold", help="data/eval.jsonl")
    s.add_argument("preds", help='{"id", "prediction"} 형식 JSONL')
    p = sub.add_parser("preview", help="미리보기 마크다운 출력")
    p.add_argument("file", nargs="?", default="data/train.jsonl")
    p.add_argument("n", nargs="?", type=int, default=30)
    p.add_argument("seed", nargs="?", type=int, default=0)
    a = ap.parse_args(argv)

    if a.cmd == "gen":
        sys.exit(1 if cmd_gen(a.train, a.eval, a.seed, a.out) else 0)
    if a.cmd == "validate":
        sys.exit(1 if cmd_validate(a.dir) else 0)
    if a.cmd == "score":
        cmd_score(a.gold, a.preds)
    if a.cmd == "preview":
        cmd_preview(a.file, a.n, a.seed)


if __name__ == "__main__":
    main()
