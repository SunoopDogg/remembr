# remembr — Claude Code 가이드

## 개발 환경
- 파일 소유자 혼재: `src/qdrant/`, `src/memory/` root 소유 → Edit 불가, 그 외 (`src/agent/` 등) airo-workstation 소유 → 직접 Edit 가능
- 수정 전 `ls -la <파일>` 으로 소유자 확인 필수
- root 소유 파일 수정: 스크립트를 `/home/airo-workstation/projects/remembr/`에 작성 후 `docker exec remembr-gpu python3 /root/remembr/<script>.py`
- Docker heredoc(`<< EOF`) 미작동 → 위 스크립트 파일 방식 사용
- Python 실행: `docker exec remembr-gpu python3 <script>`
- `ros2 pkg create`는 docker 내부 실행 → 생성 파일 전부 root 소유, 이후 수정도 스크립트 방식 필요
- `ros2 pkg create` 실행: `docker exec remembr-gpu bash -c "source /opt/ros/humble/setup.bash && cd /root/remembr/src && ros2 pkg create --build-type ament_python|ament_cmake <pkg>"`
- 새 메시지 패키지 추가 후 테스트 전 빌드 필수: `docker exec remembr-gpu bash -c "cd /root/remembr && source /opt/ros/humble/setup.bash && colcon build --packages-select <pkg>"`

## 테스트 실행
```bash
# 패키지 테스트 (qdrant, agent 등)
docker exec remembr-gpu bash -c "cd /root/remembr/src/<pkg> && source /root/remembr/install/setup.bash && python3 -m pytest test/ -p no:anyio -q"
```
- `-p no:anyio` 필수: anyio 플러그인이 pytest 6.2.5와 충돌
- `source /root/remembr/install/setup.bash` 필수: `memory_msgs`, `rclpy`, `agent_msgs` 해결

## 알려진 gotcha
- `agent/__init__.py`가 `langchain_ollama` eagerly import → `agent.services.service_factory` 단독 테스트 시 `importlib.util.spec_from_file_location`으로 직접 로드 필요
- `qdrant_client` 미설치 시 `/root/.local/bin/uv pip install qdrant-client --system`
- 컨테이너: `remembr-gpu` (메인 개발), `remembr-qdrant` (Qdrant 서버 localhost:6333)

## 린터 규칙 (src/qdrant)
- flake8 E501: 최대 줄 길이 99자
- pep257 D402: docstring 첫 줄을 함수명으로 시작 금지 (예: `def search()` → `"""Search...` 실패)

## 아키텍처
- `src/memory/` Captioner: 이미지 → Gemma 4(vLLM, port 8000) 캡션 → `CaptionWithPose` 발행
- `src/qdrant/` MemoryBuilder: Gemma 4 캡션 → Qdrant 저장 (`encode_document`)
- `src/agent/` ReMEmbRAgentNode: LLM 쿼리 루프 → Qdrant 검색 (`encode_query`)
- 임베딩: vLLM HTTP `http://localhost:8080`, 모델 `Qwen/Qwen3-Embedding-4B`
