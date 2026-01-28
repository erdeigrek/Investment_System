import streamlit as st
from pathlib import Path
from logic.quiz_engine import load_quizzes

from logic.io import AppPaths, load_yaml, default_progress_path
from logic.progress import ProgressStore

st.set_page_config(page_title="Plan nauki", page_icon="📚", layout="wide")
st.title("📚 Plan nauki — dashboard")

paths = AppPaths.from_app_file(Path(__file__))
plan_path = paths.data_dir / "learning_plan.yaml"
progress_path = default_progress_path(paths.data_dir)
store = ProgressStore(progress_path)

plan = load_yaml(plan_path)
meta = plan.get("meta", {})
epics = plan.get("epics", [])
quiz_path = paths.data_dir / "quiz.yaml"
quiz_yaml = load_yaml(quiz_path)
_, quizzes = load_quizzes(quiz_yaml)
quiz_task_ids = set(quizzes.keys())

if not epics:
    st.error("Brak 'epics' w learning_plan.yaml")
    st.stop()

st.caption(
    f"{meta.get('title', '')} | decyzja: {meta.get('decision_moment')} | target: {meta.get('target_definition')}"
)

# ---------- Helpers ----------
def flatten_tasks(epics_list):
    flat = []
    order = 0
    for e in epics_list:
        for t in e.get("tasks", []):
            flat.append(
                {
                    "order": order,
                    "epic_id": e["id"],
                    "epic_title": e["title"],
                    "task_id": t["id"],
                    "task_title": t["title"],
                    "task": t,
                }
            )
            order += 1
    return flat


def status_icon(status: str) -> str:
    return "✅" if status == "done" else ("🟡" if status == "in_progress" else "⬜")


def compute_global_stats(flat):
    counts = {"done": 0, "in_progress": 0, "todo": 0}
    for item in flat:
        s = store.get_status(item["task_id"])
        counts[s] = counts.get(s, 0) + 1
    total = len(flat)
    done = counts["done"]
    progress = (done / total) if total else 0.0
    return total, counts, progress


def next_tasks(flat, n=5):
    # najpierw in_progress, potem todo
    ordered = []
    for s in ("in_progress", "todo"):
        ordered.extend([x for x in flat if store.get_status(x["task_id"]) == s])
    return ordered[:n]


def first_next_task(flat):
    candidates = next_tasks(flat, n=1)
    return candidates[0] if candidates else None


def set_task_status(task_id: str, new_status: str):
    current = store.get_status(task_id)
    if new_status != current:
        store.set_status(task_id, new_status)



# ---------- Global progress ----------
flat = flatten_tasks(epics)

if not flat:
    st.warning("Plan nie zawiera tasków.")
    st.stop()

total, counts, progress = compute_global_stats(flat)

col1, col2, col3, col4 = st.columns([1, 1, 1, 2], gap="large")
with col1:
    st.metric("✅ Done", counts["done"])
with col2:
    st.metric("🟡 In progress", counts["in_progress"])
with col3:
    st.metric("⬜ Todo", counts["todo"])
with col4:
    st.progress(progress, text=f"Ogólny postęp: {int(progress * 100)}%  ({counts['done']}/{total})")

st.divider()

# ---------- Next step card ----------
st.subheader("➡️ Co robię teraz")

next_item = first_next_task(flat)
if next_item is None:
    st.success("Wszystko zaliczone ✅ (albo nie masz jeszcze tasków).")
else:
    task_id = next_item["task_id"]
    task = next_item["task"]
    s = store.get_status(task_id)
    icon = status_icon(s)

    with st.container(border=True):
        top_left, top_right = st.columns([3, 1], gap="large")

        with top_left:
            st.markdown(f"### {icon} {task_id} — {next_item['task_title']}")
            st.caption(f"{next_item['epic_id']} — {next_item['epic_title']}")
            # krótkie "co mam zrobić" w 3 linijkach:
            books = task.get("books", [])
            done_if = task.get("done_if", [])
            st.markdown("**Konkretnie:**")
            st.write(f"1) Przerób źródła: {books[0] if books else '—'}")
            st.write(f"2) Zrób notatkę (2–5 zdań): własnymi słowami")
            st.write(f"3) Zaliczenie: {done_if[0] if done_if else 'quiz / kryterium z planu'}")

        with top_right:
            st.markdown("**Szybkie akcje**")
            st.button("Ustaw: in_progress", key=f"btn_ip_{task_id}", on_click=set_task_status, args=(task_id, "in_progress"))
            st.button("Ustaw: done", key=f"btn_done_{task_id}", on_click=set_task_status, args=(task_id, "done"))
            st.button("Reset: todo", key=f"btn_todo_{task_id}", on_click=set_task_status, args=(task_id, "todo"))
            can_quiz = (task_id in quiz_task_ids) and (store.get_status(task_id) != "done")
            if can_quiz:
                if st.button("📝 Idź do quizu", key=f"btn_quiz_{task_id}"):
                    st.session_state["quiz_target_task_id"] = task_id
                    try:
                        st.switch_page("pages/2_Quiz.py")
                    except Exception:
                        st.warning("Nie mogę automatycznie przełączyć strony. Kliknij stronę Quiz w menu po lewej — quiz ustawi się na ten task.")
            else:
                st.caption("Brak quizu albo już zaliczone ✅")

    with st.expander("Szczegóły (źródła, kryteria, notatka)", expanded=False):
        st.markdown("**Co mam rozumieć:**")
        for u in task.get("understanding", []):
            st.write(f"- {u}")

        st.markdown("**Źródła:**")
        for b in task.get("books", []):
            st.write(f"- {b}")

        st.markdown("**Zaliczony, jeśli:**")
        for d in task.get("done_if", []):
            st.write(f"- {d}")

        st.markdown("**Notatka (Twoje wnioski / blokady):**")
        note = store.get_note(task_id)
        new_note = st.text_area(" ", value=note, height=120, key=f"note_next_{task_id}")
        if new_note != note:
            store.set_note(task_id, new_note)
            st.info("Zapisano notatkę.")

st.divider()

# ---------- Next 5 cards ----------
st.subheader("📌 Kolejne kroki (lista)")

upcoming = next_tasks(flat, n=5)
if not upcoming:
    st.info("Brak kolejnych kroków (wszystko done).")
else:
    for item in upcoming:
        task_id = item["task_id"]
        task = item["task"]
        s = store.get_status(task_id)
        icon = status_icon(s)

        with st.container(border=True):
            header_left, header_right = st.columns([3, 1], gap="large")

            with header_left:
                st.markdown(f"### {icon} {task_id} — {item['task_title']}")
                st.caption(f"{item['epic_id']} — {item['epic_title']} | status: **{s}**")

            with header_right:
                # status selector w karcie (szybko i czytelnie)
                options = ["todo", "in_progress", "done"]
                idx = options.index(s) if s in options else 0
                new_s = st.selectbox(
                    "Status",
                    options=options,
                    index=idx,
                    key=f"status_sel_{task_id}",
                    label_visibility="collapsed",
                )
                if new_s != s:
                    store.set_status(task_id, new_s)

                

            # mini instrukcja co zrobić
            books = task.get("books", [])
            done_if = task.get("done_if", [])
            st.markdown("**Konkretnie:**")
            st.write(f"- Czytaj: {books[0] if books else '—'}")
            st.write(f"- Zaliczenie: {done_if[0] if done_if else '—'}")

            with st.expander("Szczegóły", expanded=False):
                st.markdown("**Co mam rozumieć:**")
                for u in task.get("understanding", []):
                    st.write(f"- {u}")

                st.markdown("**Źródła:**")
                for b in books:
                    st.write(f"- {b}")

                st.markdown("**Zaliczony, jeśli:**")
                for d in done_if:
                    st.write(f"- {d}")

                st.markdown("**Notatka:**")
                note = store.get_note(task_id)
                new_note = st.text_area(" ", value=note, height=100, key=f"note_{task_id}")
                if new_note != note:
                    store.set_note(task_id, new_note)
                    st.info("Zapisano notatkę.")

st.divider()

# ---------- EPIC progress (important but not cluttering) ----------
with st.expander("📊 Postęp per EPIC (szczegóły)", expanded=False):
    for e in epics:
        tasks = e.get("tasks", [])
        if not tasks:
            continue
        done = sum(1 for t in tasks if store.get_status(t["id"]) == "done")
        total_e = len(tasks)
        p = done / total_e if total_e else 0.0

        st.markdown(f"**{e['id']} — {e['title']}**  ({done}/{total_e})")
        st.progress(p)

st.caption(f"Progres zapisuje się do: {progress_path}")
