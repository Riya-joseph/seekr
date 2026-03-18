# 🔍 Seekr

**Offline semantic file search for your machine.**
Find files by *what they mean*, not just what they're named.

```
seekr search "JWT authentication middleware"
seekr search "sunset over ocean" --type image
seekr search "database connection pool"
```

No cloud. No API keys. No telemetry. Fully local.

---

## Features

| Feature | Details |
|---|---|
| **Semantic search** | Context-aware search using sentence embeddings |
| **Image search** | Find images with text queries via CLIP |
| **Auto-indexing** | Watch directories and re-index on change |
| **Incremental** | SHA-256 change detection — only re-index modified files |
| **Background watch** | `--daemon` mode; logs to `~/.seekr/watch.log` |
| **Offline** | All models run locally; cached after first download |
| **Fast** | FAISS inner-product index; exact search up to ~500k chunks |
| **Multi-format** | `.txt` `.md` `.py` `.js` `.go` `.rs` `.pdf` `.jpg` `.png` + more |
| **Cross-platform** | Linux, macOS, Windows |

---

## Architecture

```
seekr/
├── domain/              # Pure Python — zero external deps
│   ├── entities.py      #   FileChunk, FileRecord, SearchResult, IndexTask, …
│   ├── interfaces.py    #   EmbeddingModel, VectorStore, FileParser, IndexQueue, …
│   └── exceptions.py    #   Domain-specific exception hierarchy
│
├── application/         # Orchestration only — no infra imports
│   ├── index_service.py #   IndexService: walk → parse → embed → store
│   ├── search_service.py#   SearchService: embed query → search → resolve
│   └── watcher_service.py#  WatcherService: watch → delegate to IndexService
│
├── config/
│   └── settings.py      # Tunable constants; all overridable via env vars
│
├── utils/
│   └── logging.py       # Centralized logging configuration
│
├── infrastructure/      # Concrete implementations
│   ├── text_embedder.py #   SentenceTransformerEmbedder (bge-small-en-v1.5)
│   ├── clip_embedder.py #   CLIPEmbedder (openai/clip-vit-base-patch32)
│   ├── faiss_store.py   #   FAISSVectorStore (IndexFlatIP + IDMap2)
│   ├── sqlite_store.py  #   SQLiteMetadataStore (WAL mode)
│   ├── parsers.py       #   PlainTextParser, CodeParser, PDFParser, ImageParser
│   ├── watcher.py       #   WatchdogFileWatcher (inotify / FSEvents / ReadDirectoryChangesW)
│   ├── ignore.py        #   Default ignore patterns + .seekrignore loading
│   ├── daemon.py        #   Background process management (POSIX fork / Windows spawn)
│   ├── container.py     #   Dependency injection factory
│   ├── queue/
│   │   └── index_queue.py  # SQLite-backed task queue
│   └── workers/
│       └── index_worker.py # Background embedding worker pool
│
└── cli/
    └── main.py          #   Typer commands — depends only on application/
```

### Dependency rule

```
cli → application → domain ← infrastructure
```

- **Domain** knows nothing about FAISS, sentence-transformers, or SQLite.
- **Application** depends only on domain interfaces (ports).
- **Infrastructure** implements those interfaces (adapters).
- **CLI** wires everything together via `Container`, then calls application services.
- **No circular dependencies.** `mypy --strict` passes.

### Two vector stores

Seekr maintains two parallel FAISS indices:

| Store | Model | Dim | Used for |
|---|---|---|---|
| `text_index/` | bge-small-en-v1.5 | 384 | Text & code retrieval |
| `clip_index/` | clip-vit-base-patch32 | 512 | Image ↔ text cross-modal |

This separation lets text search use the best text embedder while still
supporting "show me images of cats" queries.

### Background indexing queue

`seekr index` and `seekr watch` enqueue tasks in a SQLite-backed queue
(the `index_tasks` table inside `metadata.db`). A worker pool (default: 4 threads)
drains the queue while a live progress bar tracks completion. In watch mode workers
keep running between events, picking up new tasks as files change.

---

## Installation

### Linux (Ubuntu 22.04+)

```bash
sudo apt update
sudo apt install python3.10 python3-pip python3-venv git
python3 -m venv ~/.seekr-env
source ~/.seekr-env/bin/activate
git clone https://github.com/Riya-joseph/seekr.git
cd seekr
pip install -e .
```

> **GPU / CUDA users:** Seekr installs CPU-only PyTorch by default. To use a
> CUDA-enabled GPU, install the CUDA build of torch *before* installing seekr:
>
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> pip install -e ".[gpu]"   # installs faiss-gpu instead of faiss-cpu
> ```

### macOS

```bash
brew install python@3.11
python3 -m venv ~/.seekr-env
source ~/.seekr-env/bin/activate
git clone https://github.com/Riya-joseph/seekr.git
cd seekr
pip install -e .
```

### Windows

```powershell
python -m venv $env:USERPROFILE\.seekr-env
$env:USERPROFILE\.seekr-env\Scripts\Activate.ps1
git clone https://github.com/Riya-joseph/seekr.git
cd seekr
pip install -e .
```

> **Windows note:** All features work on Windows. Background watch mode
> (`--daemon`) uses `subprocess` with `DETACHED_PROCESS` flags instead of
> POSIX `fork` — no third-party tools required.

### Verify

```bash
seekr --help
```

> **First run** will download ~120 MB (bge-small) + ~600 MB (CLIP) into
> `~/.seekr/models/`. This only happens once; subsequent runs are instant.

---

## Usage

### Index a directory

**Recommendation:** Index a **project or docs directory**, not your whole home
(`~`) or root (`/`). That keeps the index small and avoids pulling in
dependencies and caches. Use [path exclusions](#path-exclusions) and
`--dry-run` to preview.

```bash
# Index a directory (defaults to current directory when --path is omitted)
seekr index
seekr index --path ~/projects/myapp
seekr index --path ~/documents

# Preview what would be indexed — no writes
seekr index --path ~/projects/myapp --dry-run

# Add extra exclusions for this run
seekr index --path ~/projects/myapp --exclude "vendor,coverage"

# Show verbose output
seekr index --path ~/documents --verbose
```

Output:
```
╭─── Index ────────────────────────────────╮
│  Scanning files…                         │
│  Discovered 431 files                    │
│  Queued 342 indexing tasks               │
╰──────────────────────────────────────────╯
[████████████████████] 342/342  100%  12s
╭─── Index ────────────────────────────────╮
│  Indexing complete.                      │
╰──────────────────────────────────────────╯
```

### Search

```bash
# Semantic text search
seekr search "database connection pool"

# Image search with text
seekr search "sunset over the ocean" --type image

# Filter by file type
seekr search "JWT token validation" --type code

# Limit results to files under a specific path
seekr search "config" --path ~/projects/myapp

# Get more results
seekr search "error handling" --top 20
```

Output:
```
╭── Results for: JWT token validation ──────────────────────────────────────────╮
│  #  Score   Type      File                                                     │
│  1  94.2%  💻 Code   ~/api/middleware.py                                       │
│                      /home/user/api/middleware.py                              │
│  2  87.1%  💻 Code   ~/auth/guards.ts                                          │
│                      /home/user/auth/guards.ts                                 │
│  3  81.3%  📄 Text   ~/docs/auth.md                                            │
│                      /home/user/docs/auth.md                                   │
╰─────────────────────── Ctrl+Click a file path to open ────────────────────────╯
```

File paths are **clickable hyperlinks** in supported terminals (GNOME Terminal
3.50+, Kitty, iTerm2, WezTerm, VS Code integrated terminal). Ctrl+Click opens
the file with your default application.

### Watch mode (continuous indexing)

```bash
# Foreground — blocks until Ctrl+C
seekr watch ~/documents
seekr watch ~/projects ~/notes ~/downloads

# Background daemon — returns to the prompt immediately
seekr watch ~/projects --daemon

# Stop the background daemon
seekr watch-stop
```

Foreground events are printed in real time:
```
09:14:22  ➕ created:  ~/notes/meeting-2025.md
09:14:45  ✏️ modified: ~/projects/api/auth.py
09:15:01  🗑️ deleted:  ~/downloads/draft.txt
```

In daemon mode all output goes to `~/.seekr/watch.log`. The watcher applies
the same path exclusions as `seekr index` and uses a background worker pool
to embed and store files asynchronously.

### Status

```bash
seekr status
```

```
╭─── Seekr Index Status ──────────────────╮
│  Data directory   ~/.seekr               │
│  Total files      1,247                  │
│  Text files       1,189                  │
│  Image files         58                  │
│  Total chunks     8,432                  │
│  Text vectors     8,374                  │
│  CLIP vectors        58                  │
│  Index size       12.4 MB                │
│  Last updated     2025-04-15 09:12 UTC   │
╰──────────────────────────────────────────╯
```

If an indexing queue has tasks, `seekr status` also shows queue progress:

```
╭─── Index Queue ─────╮
│  Total tasks    342  │
│  Completed      310  │
│  Processing       4  │
│  Pending         28  │
│  Failed           0  │
│  Progress        91% │
╰─────────────────────╯
```

### Prune and reset

Shrink or wipe the index without touching files on disk:

```bash
# Remove one project or directory from the index (un-index that subtree)
seekr prune --path ~/projects/repo-a

# Prune the current directory
seekr prune

# Wipe the entire index and start over (keeps cached models)
seekr reset
seekr reset --force   # skip confirmation
```

After **prune**, only that path's data is removed; the rest of the index is
unchanged. After **reset**, run `seekr index --path <path>` to rebuild from scratch.

---

## Index location

All data is stored in `~/.seekr/`:

```
~/.seekr/
├── models/              # Cached HuggingFace model weights
├── text_index/
│   ├── index.faiss      # Text FAISS binary index
│   └── id_map.json      # chunk_id ↔ int64 mapping
├── clip_index/
│   ├── index.faiss      # CLIP FAISS binary index
│   └── id_map.json
├── metadata.db          # SQLite: file records, chunk mappings, + indexing task queue
├── watch.pid            # PID file for the daemon watcher (when running)
├── watch.log            # Daemon watcher log
└── .seekrignore         # Global path exclusion patterns (optional)
```

Override with `--data-dir`:

```bash
seekr index --path ~/docs --data-dir /mnt/external/seekr-index
seekr search "budget" --data-dir /mnt/external/seekr-index
```

---

## Configuration

### Environment variables

All constants in `seekr/config/settings.py` can be overridden at runtime via
environment variables — no code changes needed:

| Variable | Default | Description |
|---|---|---|
| `SEEKR_MAX_CHARS_PER_FILE` | `1000000` | Max characters read per file (1 MB) |
| `SEEKR_MAX_CHUNKS_PER_FILE` | `200` | Hard cap on chunks per file |
| `SEEKR_CHUNK_SIZE` | `500` | Chunk size in token-equivalents |
| `SEEKR_CHUNK_OVERLAP` | `50` | Overlap between consecutive chunks |
| `SEEKR_NUM_INDEX_WORKERS` | `4` | Parallel background indexing threads |
| `SEEKR_TEXT_MODEL` | `BAAI/bge-small-en-v1.5` | Text embedding model |
| `SEEKR_CLIP_MODEL` | `openai/clip-vit-base-patch32` | CLIP model for images |
| `SEEKR_DEVICE` | `cpu` | Compute device: `cpu`, `cuda`, `mps` |

Example — use GPU and more workers:

```bash
SEEKR_DEVICE=cuda SEEKR_NUM_INDEX_WORKERS=8 seekr index --path ~/projects
```

### Shell alias for persistent settings

```bash
# ~/.bashrc or ~/.zshrc
alias seekr='seekr --data-dir /fast/ssd/.seekr'
```

---

## Path exclusions

To keep the index small, Seekr **always skips** a built-in list of noisy
directories when walking a tree. Users can extend this list at three levels:

| Source | Location | Use case |
|--------|----------|----------|
| **Default (always applied)** | Built-in `DEFAULT_IGNORE_PATTERNS` | See full list below |
| **Global** | `<data-dir>/.seekrignore` (default: `~/.seekr/.seekrignore`) | Your personal ignores for all runs |
| **Local** | `<index-root>/.seekrignore` | Per-project ignores, placed in the directory being indexed |
| **CLI** | `--exclude "dir1,dir2"` | One-off extra patterns for a single run |

### How matching works

Patterns are matched against **file and directory names** (not full paths). Three pattern types are supported:

| Type | Example | Matches |
|------|---------|---------|
| **Exact name** | `node_modules` | Any file or directory named exactly `node_modules` |
| **Extension** | `.csv` | Any file whose extension is `.csv` (e.g. `data.csv`) |
| **Glob** | `*.log`, `tmp_*`, `report_202?` | Any name matching the wildcard pattern |

```bash
# Exact name — skip a directory everywhere in the tree
seekr index --path ~/projects --exclude "vendor"

# Extension — skip all CSV files
seekr index --path ~/projects --exclude ".csv"

# Glob — skip all log files and any directory starting with "tmp_"
seekr index --path ~/projects --exclude "*.log,tmp_*"

# Combine all three
seekr index --path ~/projects --exclude "vendor,.csv,*.log"
```

The same syntax works in `.seekrignore` files — one pattern per line.

### Default ignore patterns

The full built-in `DEFAULT_IGNORE_PATTERNS` list:

| Category | Patterns |
|----------|----------|
| VCS | `.git`, `.svn`, `.hg`, `.bzr` |
| Python | `__pycache__`, `.venv`, `venv`, `env`, `.tox`, `.mypy_cache`, `.ruff_cache`, `.pytest_cache`, `egg-info`, `.eggs` |
| Node / frontend | `node_modules`, `.next`, `.nuxt`, `.output`, `dist`, `.parcel-cache`, `.cache`, `.turbo` |
| Rust / Go / C | `target`, `vendor`, `build`, `out` |
| IDE / editor | `.idea`, `.vscode`, `.vs` |
| OS / misc | `.DS_Store`, `Thumbs.db` |
| Build / test output | `coverage`, `.coverage`, `htmlcov`, `.nx`, `.direnv`, `tmp`, `temp`, `.tmp`, `.temp` |

All hidden directories (names starting with `.`) are also always skipped during traversal, regardless of the patterns above.

### `.seekrignore` format

One pattern per line, `#` for comments:

```
# .seekrignore
my-big-dataset
raw_exports
scratch
```

> **Note:** If you use `--data-dir` to change the data directory, the global `.seekrignore` is read from `<data-dir>/.seekrignore`, not `~/.seekr/.seekrignore`.

### Dry run

Run `seekr index --path <path> --dry-run` to preview what would actually change
before committing to a full index. The output shows:

- **New / changed files** — files that would be indexed
- **Already indexed (unchanged)** — files that would be skipped (hash unchanged)
- Estimated chunk count for the new files

Adjust `--exclude` or `.seekrignore` if the count is too high.

---

## Performance

Benchmarks on a 32 GB RAM / 8-core CPU system:

| Metric | Value |
|---|---|
| Indexing throughput | ~200 text files/sec |
| Search latency (10k chunks) | <50 ms |
| Search latency (500k chunks) | <300 ms |
| Memory at 100k chunks | ~2 GB |
| bge-small model size | 120 MB |
| CLIP model size | 600 MB |

---

## Engineering decisions

### Why bge-small-en-v1.5?

Outperforms all-MiniLM-L6-v2 on MTEB benchmarks at the same 384 dims and
similar CPU latency. Falls back gracefully to MiniLM if bge-small isn't
available.

### Why FAISS IndexFlatIP?

Exact search, zero recall loss, and fits comfortably in RAM for up to ~1M
chunks at 384 dims (≈1.5 GB). The interface is stable — swapping to
`IndexHNSWFlat` for approximate search requires changing only `faiss_store.py`.

### Why two separate vector stores?

CLIP and sentence-transformers produce incompatible embedding spaces. Mixing
them in one index would corrupt search quality. Keeping them separate allows
each model to serve its best use case.

### Why SQLite over a metadata file?

Concurrent reads, ACID transactions, WAL mode, zero setup. Perfect for
single-machine use; can be replaced with PostgreSQL for a team server.

### Chunking strategy

500 "tokens" ≈ 2000 characters, 50-token overlap ≈ 200 characters. Overlap
ensures that semantically cohesive sentences spanning a chunk boundary appear
in at least one chunk's embedding. Both values are configurable via env vars
(`SEEKR_CHUNK_SIZE`, `SEEKR_CHUNK_OVERLAP`).

### Background worker design

`seekr index` and `seekr watch --daemon` share the same SQLite queue + worker
pool. In drain mode (after `seekr index`) workers exit when the queue empties.
In watch mode workers poll the queue every 0.5 s, staying alive between file
events — 0% CPU while idle (threads park on `time.sleep`).

### Cross-platform daemon mode

- **Linux / macOS**: POSIX double-fork (`os.fork` + `os.setsid`). Parent exits;
  child redirects stdio to `watch.log` and continues as a session-leader daemon.
- **Windows**: `subprocess.Popen` with `DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
  | CREATE_NO_WINDOW` flags. Parent writes the PID file and exits; child runs
  with a hidden `--_worker` flag that skips the daemonize step.

---

## Scaling to 100k+ files

### Performance bottlenecks

1. **Exact FAISS search** — `IndexFlatIP` is O(n). At 500k+ vectors it
   becomes noticeable. **Fix**: Replace with `IndexHNSWFlat(dim, 32)` in
   `faiss_store.py` — only that file changes.

2. **Single-process embedding** — Sentence-transformers and CLIP are CPU-bound.
   **Fix**: Add a multiprocessing `ProcessPoolExecutor` in `IndexService._embed_and_store`.

3. **SQLite write contention** — One writer at a time. **Fix**: Switch to
   PostgreSQL with `asyncpg` for a team server.

4. **Model loading time** — ~2–4 seconds on cold start. **Fix**: Run Seekr
   as a daemon process with a Unix socket for CLI ↔ server IPC.

### Internal-team version

To scale to a team:

| Component | Change |
|---|---|
| `FAISSVectorStore` | Replace with Qdrant or pgvector |
| `SQLiteMetadataStore` | Replace with PostgreSQL |
| `SentenceTransformerEmbedder` | Add request batching + caching |
| `WatchdogFileWatcher` | Replace with a queue (Redis / SQS) per machine |
| CLI | Add `seekr serve` → FastAPI REST server |
| Auth | Add user-namespaced indices in the metadata store |

The domain interfaces remain **unchanged** — only adapters swap out.

---

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Lint
ruff check seekr/

# Type-check
mypy seekr/

# Tests
pytest tests/ -v --cov=seekr
```

---

## License

MIT
