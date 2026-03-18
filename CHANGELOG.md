# Changelog

All notable changes to Seekr are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Seekr uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] — 2026-03-17

### Added
- Initial release.
- Offline semantic file search with FAISS + sentence-transformers (text) and
  CLIP (images).
- `seekr index`, `seekr search`, `seekr watch`, `seekr watch-stop`,
  `seekr status`, `seekr prune`, `seekr reset` CLI commands.
- Background indexing queue backed by SQLite (`metadata.db`).
- `.seekrignore` support (global `~/.seekr/.seekrignore` and per-project).
- Daemon watch mode (`seekr watch --daemon`) on Linux, macOS, and Windows.
- Clean layered architecture: domain → application → infrastructure / CLI.
