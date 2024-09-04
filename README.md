# milvus-test

- test repository for milvus the vector database
- 스크립트를 실행하기 위해서는 docker가 필요합니다.

## environment

### run milvus (using docker-compose)

```bash
# docker-compose를 통해서 milvus-standalone을 실행시킵니다.
$ docker-compose up -d
```

```bash
# 다음의 명령어를 통해 milvus의 상태를 확인합니다.
$ docker port milvus-standalone 19530/tcp
```

### run in Slurm with HPC

- **slurm 내에서 실행하는 경우 아래와 같은 사전 절차를 거칩니다.**

```bash
# module av에 있는 프리셋 파이썬 중에서 최신 버전을 가져옵니다.
$ module add python/3.11.2
```

```bash
# cpu1 서버에 직접 접속하여 특정 작업을 실행시킵니다. (cpu1 파티션, 48코어)
$ srun --pty -p cpu1 -c 48 /bin/bash
```

```bash
# milvus 데이터베이스의 주소를 .env 파일에 집어넣습니다. (192.168.0.1이라고 가정)
$ echo "DATABASE_URL=192.168.0.1" > .env
```

### run python (using venv)

- **가상환경을 작동시킨 후 패키지를 설치합니다. (최초 1회에 한하여 가상환경을 설정합니다)**

```bash
# 가상 환경 생성
$ python3 -m venv venv
```

```bash
# 가상 환경 활성화
$ source venv/bin/activate
```

```bash
# requirement.txt를 통한 패키지 설치
$ pip3 install -r requirements.txt
```

- **가상환경을 종료하거나, 패키지 설정을 저장하는 경우**

```bash
# 패키지 의존성 저장
$ pip3 freeze > requirements.txt
```

```bash
# 가상 환경 비활성화
$ deactivate
```

```bash
# !!! 패키지 의존성 초기화 !!!
$ pip3 uninstall -r requirements.txt -y
```
