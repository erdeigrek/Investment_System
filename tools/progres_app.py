from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import streamlit as st


@dataclass
class Task:
    line_idx: int
    checked: bool
    text: str
    section: str


CHECKBOX_RE = re.compile(r"^(\s*)-\s*\[(x| )\]\s+(.*)$", re.IGNORECASE)
SECTION_RE = re.compile(r"^\s*##\s+(.*)$")


def parse_todo(md_text: str) -> tuple[list[str], list[Task], list[str]]:
    """
    Returns:
      - lines: original lines
      - tasks: parsed checkbox tasks with section info
      - sections: ordered list of section names encountered
    """
    lines = md_text.splitlines()
    tasks: list[Task] = []
    sections: list[str] = []

    current_section = "Bez sekcji"

    for i, line in enumerate(lines):
        sec = SECTION_RE.match(line)
        if sec:
            current_section = sec.group(1).strip()
            if current_section not in sections:
                sections.append(current_section)
            continue

        m = CHECKBOX_RE.match(line)
        if m:
            checked = m.group(2).lower() == "x"
            text = m.group(3).strip()
            tasks.append(Task(line_idx=i, checked=checked, text=text, section=current_section))

    return lines, tasks, sections


def set_task_checked(lines: list[str], task: Task, checked: bool) -> None:
    """Mutates lines to update checkbox state at task.line_idx."""
    line = lines[task.line_idx]
    m = CHECKBOX_RE.match(line)
    if not m:
        return
    indent = m.group(1)
    box = "x" if checked else " "
    text = m.group(3)
    lines[task.line_idx] = f"{indent}- [{box}] {text}"


def main() -> None:
    st.set_page_config(page_title="Monitor nauki", layout="wide")
    st.title("🧠 Monitor nauki — Investment System")

    root = Path.cwd()
    todo_path = root / "TODO.md"

    st.sidebar.header("Plik")
    st.sidebar.code(str(todo_path), language="text")

    if not todo_path.exists():
        st.error("Brak TODO.md w katalogu projektu. Utwórz plik TODO.md w root.")
        st.stop()

    md_text = todo_path.read_text(encoding="utf-8")
    lines, tasks, sections = parse_todo(md_text)

    total = len(tasks)
    done = sum(1 for t in tasks if t.checked)
    pct = (done / total) if total else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Zrobione", f"{done}")
    c2.metric("Wszystkie", f"{total}")
    c3.metric("Progres", f"{pct*100:.1f}%")
    st.progress(pct)

    st.divider()

    # Filters
    st.sidebar.header("Filtry")
    only_open = st.sidebar.checkbox("Pokaż tylko niezrobione", value=False)
    section_filter = st.sidebar.selectbox("Sekcja", ["(wszystkie)"] + sections)

    # Group tasks by section
    by_section: dict[str, list[Task]] = {}
    for t in tasks:
        by_section.setdefault(t.section, []).append(t)

    changed = False

    for sec_name, sec_tasks in by_section.items():
        if section_filter != "(wszystkie)" and sec_name != section_filter:
            continue

        # section stats
        sec_total = len(sec_tasks)
        sec_done = sum(1 for t in sec_tasks if t.checked)
        with st.expander(f"{sec_name} — {sec_done}/{sec_total}", expanded=True):
            for t in sec_tasks:
                if only_open and t.checked:
                    continue
                key = f"task_{t.line_idx}"
                new_val = st.checkbox(t.text, value=t.checked, key=key)
                if new_val != t.checked:
                    set_task_checked(lines, t, new_val)
                    t.checked = new_val
                    changed = True

    st.divider()

    col_a, col_b = st.columns([1, 2])
    with col_a:
        if st.button("💾 Zapisz zmiany do TODO.md", disabled=not changed):
            todo_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            st.success("Zapisano TODO.md")
    with col_b:
        st.caption("Tip: dopisuj własne taski w TODO.md w formacie `- [ ] ...` w dowolnej sekcji `## ...`.")

    st.subheader("Podgląd TODO.md")
    st.markdown(todo_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
