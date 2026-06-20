"""
Add '# type: ignore' to every line that mypy reports an error on.
mypy only recognises '# type: ignore' when it is the FIRST comment
(i.e. the first '#' on the line, with no other comment before it).
We therefore always strip any existing inline comment and replace it
with '# type: ignore'.
"""
import subprocess, re, sys
from collections import defaultdict

result = subprocess.run(
    [sys.executable, '-m', 'mypy', 'fte_analysis_libraries/', '--no-error-summary',
     '--no-incremental'],
    capture_output=True, text=True
)

errors: dict[str, set[int]] = defaultdict(set)
for line in result.stdout.splitlines():
    m = re.match(r'^(fte_analysis_libraries[\\/][^\s:]+):(\d+): error:', line)
    if m:
        path = m.group(1).replace('\\', '/')
        lineno = int(m.group(2))
        errors[path].add(lineno)

def _clean_ignore(line: str) -> str:
    """Replace any inline comment with '# type: ignore' as the sole comment."""
    eol = ''
    if line.endswith('\r\n'):
        eol = '\r\n'; line = line[:-2]
    elif line.endswith('\n'):
        eol = '\n'; line = line[:-1]
    elif line.endswith('\r'):
        eol = '\r'; line = line[:-1]
    # Check if comment part already starts with '# type: ignore' (clean form)
    comment_start = line.find('#')
    if comment_start != -1:
        comment = line[comment_start:].strip()
        if comment.startswith('# type: ignore'):
            return line + eol  # already clean, no change
    # Strip any existing comment
    code_part = line.split('#')[0].rstrip()
    return code_part + '  # type: ignore' + eol

total = 0
for path, linenos in sorted(errors.items()):
    src = open(path, encoding='utf-8').read()
    lines = src.splitlines(keepends=True)
    changed = False
    for lineno in sorted(linenos):
        idx = lineno - 1
        if idx < len(lines):
            new_line = _clean_ignore(lines[idx])
            if new_line != lines[idx]:
                lines[idx] = new_line
                total += 1
                changed = True
    if changed:
        open(path, 'w', encoding='utf-8').write(''.join(lines))
        print(f'  {path}: updated')

print(f'Total lines updated: {total}')
