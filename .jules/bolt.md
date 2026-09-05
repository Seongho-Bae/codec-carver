## 2024-05-28 - Avoid O(N^2) Path.resolve() in Batch Processing
**Learning:** Python's `pathlib.Path.resolve()` is relatively slow because it touches the filesystem to follow symlinks and resolve relative paths. When dealing with a batch operation (e.g., scanning large directories of media files), calculating protected files via `any(target == src.resolve() for src in sources)` on every check leads to massive O(N^2) CPU overhead.
**Action:** Pre-resolve the entire list of candidate paths once into a `frozenset` at the beginning of the batch process. Pass this resolved set down the call stack so that collision/protection checks become O(1) hash map lookups instead of triggering millions of unnecessary disk access operations.

## 2024-05-29 - [Optimization of Directory Traversal using os.walk]
**Learning:** `Path.rglob()` loads the entire file tree recursively without allowing pruning. This is computationally expensive, especially when filtering excludes entire directories and deep folder paths.
**Action:** Replace `rglob` with `os.walk`, modifying `dirnames` in-place to prune excluded folders directly, resulting in orders of magnitude speedups depending on search depth and amount of un-traversable folders.

## 2024-05-29 - [Unit Test Add: _first_int in media_shrinker]
**Learning:** Even simple utility functions like `_first_int` benefit from explicit tests of edge cases, particularly their exception handling blocks which often go uncovered. Placing tests before `if __name__ == "__main__":` ensures compatibility with all test execution methods.
**Action:** Always verify test file structure when appending new test classes to ensure they run correctly within the target test framework and script execution paradigms.

## 2024-05-15 - Unsafe Path Resolution Optimization
**Learning:** `Path.is_symlink()` only checks if the final path component is a symlink, missing symlinks in parent directories. Using it to conditionally skip `Path.resolve()` is dangerous and can break collision detection logic.
**Action:** Always prefer resolving paths fully or rely on caller-provided pre-resolved data structures rather than trying to build conditional bypasses for `Path.resolve()`.

## 2026-05-30 - [Optimize file size discovery to prevent redundant disk I/O]
**Learning:** Calling `stat()` multiple times per file for file size in a large directory tree is inefficient. Passing down the size determined during the candidate gathering phase skips redundant I/O operations and speeds up the entire media shrinking run when evaluating thousands of files.
**Action:** When walking directory trees and checking file sizes to filter out targets, capture and propagate these sizes in memory if downstream functions need them, instead of statting files a second time.

## 2024-05-30 - Converting CLI tool to MCP and SaaS Web Service
**Learning:** When building FastAPI apps that wrap heavy, blocking synchronous tasks (like audio/video conversion using subprocesses), do NOT use `async def` for the endpoint function. Using a synchronous `def` allows FastAPI to run the blocking task in a threadpool, preventing the event loop from stalling. Also, handle file uploads cleanly with streaming (`shutil.copyfileobj(file.file, f)`) instead of `await file.read()` to avoid OOM issues on large media files.
**Action:** Always evaluate whether wrapped library functions block I/O. If they do, expose them via synchronous `def` route handlers in FastAPI. Use `shutil.copyfileobj` for large file uploads.

## 2024-05-31 - [Avoid instantiating pathlib.Path in tight traversal loops]
**Learning:** `pathlib.Path` instantiation is relatively slow due to the internal magic and operations performed (even for simple paths). When walking large directory trees (like with `os.walk`), instantiating a `Path` for every file simply to check its suffix leads to significant overhead (often 40-50% of the traversal time).
**Action:** When filtering files by extension during discovery, use string methods like `filename.lower().endswith(tuple_of_extensions)` or `os.path.splitext` before creating the `Path` object. Only create `Path` objects for files that pass the initial string filter.
## 2026-06-08 - [Avoid excessive Path instantiation in hot loops]
**Learning:** Using `pathlib.Path` objects inside heavy directory traversal loops (like `os.walk`) causes massive performance overhead due to repeated object instantiation and system call abstraction. Even seemingly innocent methods like `is_symlink()` or `stat()` inside `Path` add up over thousands of files.
**Action:** When scanning large directories for thousands of items, prefer raw string manipulation with `os.path.join()` and direct system calls like `os.lstat()` or `os.walk(str(root))`. Convert to `Path` objects only at the very edges of the API, returning to callers.
## 2024-06-09 - Optimize Path Exclusion Checks
**Learning:** Using `any()` with generator expressions for prefix matching is slow. Python's `str.startswith()` can accept a tuple of strings directly, pushing the loop to optimized C code.
**Action:** Use `str.startswith(tuple_of_prefixes)` and `frozenset` for O(1) exact matches instead of O(N) loops.
## 2024-06-11 - [Optimize find_candidates batch gathering]
**Learning:** During batch directory traversal (e.g. `os.walk` in `find_candidates`), attempting to resolve every directory with `Path.resolve()` introduces heavy filesystem access and instantiation overhead, especially when no directories are actually excluded (`exclude_paths=[]`).
**Action:** Replaced `Path.resolve()` with `os.path.realpath` inside the hot loop which avoids object creation overhead, and guarded the entire directory exclusion check block with `if excluded_exact_strs:` so the fast path completely skips redundant realpath and stat calls when no `exclude_paths` are configured.
## 2026-06-12 - Fast Path for Regex Log Parsing
**Learning:** Regex execution in Python, even when compiled, is slower than a simple substring  check. When parsing massive logs (like FFmpeg stderr) where 99% of lines are irrelevant progress updates, adding a fast-path substring check before the regex significantly boosts performance.
**Action:** Use substring checks as a fast filter before executing regular expressions when processing large text streams where matches are infrequent.
## 2026-06-12 - Fast Path for Regex Log Parsing
**Learning:** Regex execution in Python, even when compiled, is slower than a simple substring `in` check. When parsing massive logs (like FFmpeg stderr) where 99% of lines are irrelevant progress updates, adding a fast-path substring check before the regex significantly boosts performance.
**Action:** Use substring checks as a fast filter before executing regular expressions when processing large text streams where matches are infrequent.

## 2026-06-14 - FFprobe JSON Format Optimization
**Learning:** When retrieving size during probe parsing, FFprobe includes `format.size` in the JSON payload. Relying on this avoids the heavy system calls introduced by `Path.stat()`.
**Action:** Use size values provided by ffprobe JSON parsing directly before falling back to system stat calls.

## 2024-06-15 - Fast Path Sorting using removeprefix
**Learning:** For performance-critical path sorting or manipulation of large lists in Python, using `Path.relative_to()` or `os.path.relpath()` inside `lambda`s/loops incurs heavy overhead. `Path.relative_to` creates new `Path` instances each time, and `os.path.relpath()` triggers `os.getcwd()` system calls if absolute paths aren't guaranteed.
**Action:** Pre-compute the root path as a string (e.g., `root.as_posix()`) with a trailing slash if necessary, and use string operations like `item[0].as_posix().removeprefix(root_prefix).casefold()`. This avoids object instantiation and system calls during sorting, accelerating $O(N \log N)$ operations substantially.
## 2024-05-24 - Batch Processing Path Checks
**Learning:** Instantiating `frozenset` objects via union operations (e.g., `frozenset(large_set | {item})`) inside tight inner loops causes severe O(N^2) overhead, especially when `large_set` contains thousands of path strings. Passing the original set unmodified and checking individual items independently is far more performant and requires only a constant time overhead per iteration.
**Action:** When filtering or checking exclusions inside batch process loops, do not modify or clone large collections per iteration. Always use the base set directly and check item equality independently if needed.
## 2026-06-21 - Cache stat() results for media files to avoid repeated system calls
**Learning:** Checking the size of intermediate file outputs and probing them with ffprobe results in redundant `stat()` calls if the file size can just be supplied to the probe mechanism.
**Action:** Always attempt to pass down already computed `os.stat` or `pathlib.Path.stat()` results, especially `st_size`, to child operations in batch loops instead of repeating the system call.
## 2024-06-22 - Optimize FFprobe payload parsing with single pass iteration
**Learning:** Using multiple generator expressions (`next` and `any`) to search through the same list (like FFprobe streams) requires iterating through the list multiple times. In `_parse_probe_payload`, parsing out both the audio stream and checking for a video stream with separate generator expressions introduces unnecessary loop overhead, which is measurable in batch processes.
**Action:** Combine multiple searches over the same list into a single standard `for` loop, extracting all necessary information in one pass. This provides measurable CPU savings and avoids multiple iterator instantiations.
## 2026-06-25 - [Optimize Path.exists() when paired with stat()]
**Learning:** Checking `Path.exists()` before `Path.stat()` introduces a redundant system call because `exists()` internally uses `stat()`.
**Action:** Rely on catching the `OSError` from `Path.stat()` to simultaneously check for existence and retrieve file attributes, saving measurable I/O overhead on large filesystems.
## 2026-06-12 - [Optimize FFmpeg Log Parsing with finditer]
**Learning:** When parsing large text logs from command-line tools like ffmpeg (which use carriage returns \r for progress updates), using str.splitlines() causes massive memory allocations and manual str.find() slicing fails on \r boundaries.
**Action:** Compile a unified regex and use re.finditer() directly on the raw output string for safe, memory-efficient parsing without instantiating list of lines.

## 2025-02-12 - [Fast Path Execution in Directory Traversal and Log Parsing]
**Learning:** Checking for string existence (`if "silence_" not in stderr`) before invoking regex matchers provides significant speed improvements when parsing large blocks of text. Similarly, moving expensive I/O operations like `os.path.realpath` inside conditional blocks prevents redundant disk access when configuration (like path exclusions) isn't utilized.
**Action:** When working on large text processing or disk operations, verify if early exit conditions or conditional execution can bypass the expensive system or library calls.
