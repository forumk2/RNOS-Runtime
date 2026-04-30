(
echo ===== FILE TREE =====
tree /F

echo.
echo ===== SOURCE FILES =====

for /r %%f in (*.cpp *.h *.hpp *.glsl *.vert *.frag *.compute *.cmake *.txt *.md) do (
    echo.
    echo ===== FILE: %%f =====
    type "%%f"
)
) > repo_dump.txt