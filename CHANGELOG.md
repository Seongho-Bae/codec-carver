# Changelog

## [Unreleased]
### Added
- 다중 파일 업로드 선택 시 즉각적인 파일 개수 피드백 및 제한 초과 경고 메시지 추가
- 일괄 업로드 폼에 대상 바이트 프리셋 버튼과 총 파일 크기 미리보기를 추가하여 사용성을 개선했습니다.
- 클라이언트 측 폼 검증 시 하드코딩된 '5 GiB' 텍스트를 동적으로 변환되도록 수정하고 일괄 업로드 폼에 최대 크기(MAX_UPLOAD_BYTES) 검증 피드백을 추가했습니다.

### Changed
- 순수 영숫자 토큰은 정규식 호출을 건너뛰되 다국어·문장부호 토큰화 결과는 기존 의미와 동일하게 유지합니다. 근거, 한계, APA 7 참고문헌은 [`docs/doctoring/token-fast-path-equivalence.md`](docs/doctoring/token-fast-path-equivalence.md)에 기록했습니다.

### Fixed
- 단일·일괄 대상 크기 입력을 비웠을 때 이전 custom validity와 `aria-invalid` 상태를 즉시 초기화해 현재 필수 입력 상태를 정확히 전달합니다.
- 업로드 파일명의 경로 구분자를 정규화하여 POSIX에서도 Windows 형식의 클라이언트 경로가 일관된 basename으로 기록되도록 수정했습니다.

### Changed
- 데이터베이스 연결 시마다 불필요하게 실행되던 `PRAGMA journal_mode=WAL`을 데이터베이스 초기화 시 한 번만 실행하도록 수정하여 성능을 개선했습니다.
