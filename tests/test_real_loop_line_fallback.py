import shutil
from pathlib import Path

from agent_runtime.real_loop.file_tools import FileTools
from agent_runtime.real_loop.patcher import LineEdit, convert_patch_to_line_edits


def test_convert_patch_to_line_edits_handles_anchor_mismatch_and_crlf():
    patch = "--- a/app.py\n+++ b/app.py\n@@ -10,1 +10,1 @@\n-old\n+new\n"
    edits = convert_patch_to_line_edits(patch, "a\r\nold\r\nz\r\n", target="app.py")

    assert edits == [LineEdit(line_number=2, content="new")]


def test_apply_line_edits_is_bounded_to_existing_sandbox_file():
    root = Path(".tmp_line_fallback_test")
    sandbox = root / "sandbox_repo"
    try:
        sandbox.mkdir(parents=True)
        (sandbox / "app.py").write_text("a\nold\nz\n", encoding="utf-8")
        tools = FileTools(sandbox)

        report = tools.apply_line_edits("app.py", [LineEdit(line_number=2, content="new")])

        assert report.files_modified == ("app.py",)
        assert report.lines_changed == 1
        assert tools.read_file("app.py").splitlines() == ["a", "new", "z"]
    finally:
        shutil.rmtree(root, ignore_errors=True)
