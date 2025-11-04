import asyncio
import os
import sys
import re
from pathlib import Path
from typing import List

# ------------------------------------------------------------
# Load env (tokens should live in your .env, NOT hardcoded)
# ------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv()

"""
File Scanner Agent Demo

This agent uses TinyLlama to analyze directories and identify files that may need
to be deleted based on various criteria (duplicates, large files, temp files, etc.).
"""

from fairlib import (
    HuggingFaceAdapter,
    ToolRegistry,
    AbstractTool,
    ToolExecutor,
    WorkingMemory,
    SimpleAgent,
    SimpleReActPlanner,
    RoleDefinition
)


# -----------------------------
# Path resolution helper (NEW)
# -----------------------------
def _resolve_scan_path(raw_path: str) -> Path:
    """
    Normalize user-provided paths:
    - Accept aliases like 'downloads', 'download', 'dl'
    - Expand ~ and %ENVVARS%
    - Try OneDrive/Downloads on Windows
    """
    if not raw_path:
        return Path.cwd()

    alias = raw_path.strip().strip('"').strip("'").lower()
    if alias in {"downloads", "download", "dl"}:
        home = Path.home()
        candidates = [
            home / "Downloads",
            home / "downloads",
        ]
        # Windows OneDrive variants
        if os.name == "nt":
            userprofile = os.environ.get("USERPROFILE", "")
            if userprofile:
                candidates += [
                    Path(userprofile) / "Downloads",
                    Path(userprofile) / "OneDrive" / "Downloads",
                ]
        for c in candidates:
            if c.exists() and c.is_dir():
                return c
        # Fall back to expected location even if missing (caller will error nicely)
        return home / "Downloads"

    # Expand ~ and env vars; allow forward slashes on Windows
    expanded = os.path.expanduser(os.path.expandvars(raw_path.strip('"').strip("'")))
    return Path(expanded).resolve()


class FileScannerTool(AbstractTool):
    """
    A tool that scans directories and provides information about files
    that might be candidates for deletion.
    """

    def __init__(self):
        super().__init__()
        self.name = "file_scanner"
        self.description = (
            "Scans a directory and identifies files that may need deletion. "
            "Parameters: 'path' (directory to scan), 'min_size_mb' (minimum file size in MB), "
            "'extensions' (comma-separated list like '.tmp,.log'), "
            "'scan_type' (options: 'large_files', 'temp_files', 'old_files', 'duplicates')"
        )

    def use(self, *args, **kwargs) -> str:
        """
        Scans directory based on specified criteria.
        Flexibly accepts arguments in multiple formats.
        """
        import json

        # -------- helper: parse JSON that may have unescaped Windows backslashes --------
        def _try_parse_json_with_backslash_fix(s: str):
            """
            Try json.loads. If it fails and the string looks like JSON that includes
            Windows paths with unescaped backslashes, auto-escape them.
            """
            try:
                return json.loads(s), None
            except json.JSONDecodeError as e1:
                looks_json = s.strip().startswith("{") and ("\"path\"" in s or "'path'" in s)
                has_win_drive = re.search(r'["\']path["\']\s*:\s*["\'][A-Za-z]:\\', s) is not None
                if looks_json and has_win_drive:
                    # Only target the value of "path": "..."
                    def esc_backslashes(m):
                        quote = m.group(1)
                        body = m.group(2).replace("\\", "\\\\")
                        return f"\"path\":{quote}{body}{quote}"

                    s_fixed = re.sub(r'"path"\s*:\s*([\'"])(.+?)\1', esc_backslashes, s)
                    try:
                        return json.loads(s_fixed), "auto_escaped_backslashes"
                    except json.JSONDecodeError as e2:
                        return None, f"json_error_after_fix: {e2}"
                return None, f"json_error: {e1}"

        # -----------------------------
        # Input parsing (HARDENED)
        # -----------------------------
        if args and isinstance(args[0], str):
            tool_input = args[0]

            # 1) Try JSON (with auto-escape fallback for Windows paths)
            parsed, json_status = _try_parse_json_with_backslash_fix(tool_input)
            if parsed is not None and isinstance(parsed, dict):
                path = parsed.get('path', '.')
                min_size_mb = float(parsed.get('min_size_mb', 10.0))
                extensions = parsed.get('extensions', '')
                scan_type = parsed.get('scan_type', 'large_files')
            else:
                # 2) Fallback: comma-separated inputs
                #    Examples:
                #       ~/Downloads, .zip,.iso, large_files
                #    or a brittle JSON-like first token; peel {"path":"..."} out if present
                parts = [p.strip() for p in tool_input.split(',')]
                raw = parts[0] if parts else '.'

                m = re.search(r'"path"\s*:\s*([\'"])(.+?)\1', raw)
                if m:
                    raw = m.group(2)

                path = raw
                extensions = parts[1] if len(parts) > 1 else ''
                scan_type = parts[2] if len(parts) > 2 else 'large_files'
                min_size_mb = 10.0

        elif kwargs:
            # Direct keyword arguments
            path = kwargs.get('path', '.')
            min_size_mb = float(kwargs.get('min_size_mb', 10.0))
            extensions = kwargs.get('extensions', '')
            scan_type = kwargs.get('scan_type', 'large_files')
        else:
            path = '.'
            min_size_mb = 10.0
            extensions = ''
            scan_type = 'large_files'

        # -----------------------------
        # Resolve path & normalize
        # -----------------------------
        try:
            scan_path = _resolve_scan_path(path)

            # Helpful hint for Windows JSON escaping
            hint = None
            if os.name == "nt":
                # If there's a single backslash usage in JSON, it's likely not escaped
                # e.g., "C:\Users\Luke\Downloads" (bad) vs "C:\\Users\\Luke\\Downloads" (good)
                if "\\" in (path or "") and "\\\\" not in (path or ""):
                    hint = ("On Windows JSON strings, use double backslashes "
                            "(e.g., C:\\\\Users\\\\Luke\\\\Downloads) or forward slashes (C:/Users/Luke/Downloads).")

            if not scan_path.exists():
                return json.dumps({
                    "error": f"Path '{path}' does not exist after expansion.",
                    "resolved_path": str(scan_path),
                    "hint": hint or "Try an absolute path, forward slashes, or an alias like 'downloads'."
                })
            if not scan_path.is_dir():
                return json.dumps({
                    "error": f"Path '{path}' is not a directory.",
                    "resolved_path": str(scan_path)
                })

            # Normalize extensions
            if extensions:
                ext_list: List[str] = []
                for e in extensions.split(","):
                    e = e.strip().lower()
                    if not e:
                        continue
                    if not e.startswith("."):
                        e = "." + e
                    ext_list.append(e)
            else:
                ext_list = []

            # Run selected scan
            if scan_type == "large_files":
                results, total_size = self._scan_large_files(scan_path, min_size_mb, ext_list)
            elif scan_type == "temp_files":
                results, total_size = self._scan_temp_files(scan_path)
            elif scan_type == "old_files":
                results, total_size = self._scan_old_files(scan_path, days=180)
            elif scan_type == "duplicates":
                results, total_size = self._scan_duplicates(scan_path)
            else:
                return json.dumps({"error": f"Unknown scan_type: {scan_type}"})

            return json.dumps({
                "scan_type": scan_type,
                "path": str(scan_path),
                "files_found": len(results),
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "files": results[:20],  # Limit to 20 files for readability
                "note": "Limited to first 20 files" if len(results) > 20 else "All files shown"
            })

        except Exception as e:
            import json
            return json.dumps({"error": f"Scan failed: {str(e)}"})

    def _scan_large_files(self, path: Path, min_size_mb: float, extensions: List[str]) -> tuple:
        """Find files larger than specified size."""
        results = []
        total_size = 0
        min_bytes = min_size_mb * 1024 * 1024

        for file_path in path.rglob("*"):
            if file_path.is_file():
                try:
                    size = file_path.stat().st_size
                    if size >= min_bytes:
                        if not extensions or file_path.suffix.lower() in extensions:
                            results.append({
                                "path": str(file_path.relative_to(path)),
                                "size_mb": round(size / (1024 * 1024), 2),
                                "reason": "Large file"
                            })
                            total_size += size
                except (PermissionError, OSError):
                    continue

        results.sort(key=lambda x: x["size_mb"], reverse=True)
        return results, total_size

    def _scan_temp_files(self, path: Path) -> tuple:
        """Find temporary files."""
        temp_extensions = ['.tmp', '.temp', '.cache', '.log', '.bak', '~']
        temp_patterns = ['tmp', 'temp', 'cache', '__pycache__']
        results = []
        total_size = 0

        for file_path in path.rglob("*"):
            if file_path.is_file():
                try:
                    is_temp = (
                        file_path.suffix.lower() in temp_extensions or
                        any(pattern in file_path.name.lower() for pattern in temp_patterns)
                    )
                    if is_temp:
                        size = file_path.stat().st_size
                        results.append({
                            "path": str(file_path.relative_to(path)),
                            "size_mb": round(size / (1024 * 1024), 2),
                            "reason": "Temporary file"
                        })
                        total_size += size
                except (PermissionError, OSError):
                    continue

        return results, total_size

    def _scan_old_files(self, path: Path, days: int = 180) -> tuple:
        """Find files not modified in specified days."""
        import time
        results = []
        total_size = 0
        cutoff_time = time.time() - (days * 24 * 60 * 60)

        for file_path in path.rglob("*"):
            if file_path.is_file():
                try:
                    mtime = file_path.stat().st_mtime
                    if mtime < cutoff_time:
                        size = file_path.stat().st_size
                        age_days = int((time.time() - mtime) / (24 * 60 * 60))
                        results.append({
                            "path": str(file_path.relative_to(path)),
                            "size_mb": round(size / (1024 * 1024), 2),
                            "reason": f"Not modified in {age_days} days"
                        })
                        total_size += size
                except (PermissionError, OSError):
                    continue

        return results, total_size

    def _scan_duplicates(self, path: Path) -> tuple:
        """Find potential duplicate files based on size and name similarity."""
        size_groups = {}
        results = []
        total_size = 0

        # Group files by size
        for file_path in path.rglob("*"):
            if file_path.is_file():
                try:
                    size = file_path.stat().st_size
                    size_groups.setdefault(size, []).append(file_path)
                except (PermissionError, OSError):
                    continue

        # Check files with same size for potential duplicates
        for size, files in size_groups.items():
            if len(files) > 1 and size > 0:
                for file_path in files[1:]:  # Keep first, mark others
                    results.append({
                        "path": str(file_path.relative_to(path)),
                        "size_mb": round(size / (1024 * 1024), 2),
                        "reason": f"Potential duplicate ({len(files)} files with same size)"
                    })
                    total_size += size

        return results, total_size


async def main():
    """
    Main function to assemble and run the file scanner agent.
    """
    print("🔍 Initializing File Scanner Agent with OpenAI GPT-4o-mini...")

    # Get OpenAI API key (from .env)
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("❌ Error: OPENAI_API_KEY not found in .env file!")
        print("Please add it to your .env file, e.g.:")
        print("OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        return

    # Initialize OpenAI model (faster than local models)
    from fairlib import OpenAIAdapter
    llm = OpenAIAdapter(
        api_key=openai_key,
        model_name="gpt-4o-mini"  # Fast and cheap
    )

    # Set up tool registry with file scanner
    tool_registry = ToolRegistry()
    file_scanner = FileScannerTool()
    tool_registry.register_tool(file_scanner)

    print(f"✅ Tools available: {[tool.name for tool in tool_registry.get_all_tools().values()]}")

    # Create tool executor and memory
    executor = ToolExecutor(tool_registry)
    memory = WorkingMemory()

    # Create planner (using ReActPlanner)
    from fairlib import ReActPlanner
    planner = ReActPlanner(llm, tool_registry)
    planner.prompt_builder.role_definition = RoleDefinition(
        "You are a file system analyzer assistant. Your job is to help users identify files "
        "that may need to be deleted based on various criteria like size, age, type, or duplicates. "
        "You analyze directories and provide clear recommendations. You reason step-by-step and "
        "use the file_scanner tool to gather information. "
        "When calling file_scanner, prefer forward slashes in Windows paths (e.g., C:/Users/Luke/Downloads) "
        "or double-escape backslashes in JSON (C:\\\\Users\\\\Luke\\\\Downloads)."
    )

    # Assemble the agent
    agent = SimpleAgent(
        llm=llm,
        planner=planner,
        tool_executor=executor,
        memory=memory,
        max_steps=10
    )

    print("\n✨ Agent ready! Ask me to scan directories for files to delete.")
    print("📝 Examples:")
    print("  - 'Scan the current directory for large files over 50MB'")
    print("  - 'Find all temporary files in downloads'")
    print("  - 'Look for old files not modified in 6 months'")
    print("  - 'Check for duplicate files'")
    print("\nType 'exit' to quit.\n")

    # Quick sanity check (NEW): show what 'downloads' resolves to
    try:
        print("🔎 Resolved 'downloads' to:", _resolve_scan_path("downloads"))
    except Exception as _e:
        print("⚠️ Could not resolve 'downloads':", _e)

    # Interaction loop
    while True:
        try:
            user_input = input("👤 You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("🤖 Agent: Goodbye!")
                break

            agent_response = await agent.arun(user_input)
            print(f"\n🤖 Agent: {agent_response}\n")

        except KeyboardInterrupt:
            print("\n🤖 Agent: Exiting...")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
