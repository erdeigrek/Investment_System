import streamlit as st
from pathlib import Path

from logic.io import AppPaths, load_yaml, default_progress_path
from logic.progress import ProgressStore
from logic.quiz_engine import load_quizzes, grade_single_choice

st.set_page_config(page_title="Quiz", page_icon="📝", layout="wide")
st.title("📝 Quiz")

paths = AppPaths.from_app_file(Path(__file__))
plan_path = paths.data_dir / "learning_plan.yaml"
quiz_path = paths.data_dir / "quiz.yaml"
store = ProgressStore(default_progress_path(paths.data_dir))

plan = load_yaml(plan_path)
quiz_yaml = load_yaml(quiz_path)
threshold, quizzes = load_quizzes(quiz_yaml)

epics = plan.get("epics", [])
if not epics:
    st.error("Brak 'epics' w learning_plan.yaml")
    st.stop()

# Flatten tasks for selection
flat = []
for e in epics:
    for t in e.get("tasks", []):
        flat.append((e, t))

if not flat:
    st.error("Brak tasków w learning_plan.yaml")
    st.stop()

# --- TYLKO quizy niezaliczone (exist in quiz.yaml AND status != done) ---
not_done_quiz_indices = []
not_done_labels = []

for i, (e, t) in enumerate(flat):
    tid = t["id"]
    if tid in quizzes and store.get_status(tid) != "done":
        not_done_quiz_indices.append(i)
        not_done_labels.append(f"{e['id']}/{tid} — {t['title']}")

if not not_done_quiz_indices:
    st.success("Wszystkie quizy zaliczone ✅")
    st.stop()


def local_index_for_task_id(task_id: str) -> int | None:
    """Zwraca indeks w liście niezaliczonych quizów (0..N-1), albo None."""
    for j, i in enumerate(not_done_quiz_indices):
        _, t = flat[i]
        if t["id"] == task_id:
            return j
    return None


def next_local_index(current_local: int) -> int | None:
    """Następny quiz w niezaliczonych (zawija)."""
    n = len(not_done_quiz_indices)
    if n <= 1:
        return None
    return (current_local + 1) % n


# ---- state init (PRZED widgetami!) ----
if "quiz_pick_local" not in st.session_state:
    st.session_state["quiz_pick_local"] = 0

if "last_passed" not in st.session_state:
    st.session_state["last_passed"] = False

if "last_task_id" not in st.session_state:
    st.session_state["last_task_id"] = None

# 1) skok z Planu: quiz_target_task_id ma pierwszeństwo
target_task_id = st.session_state.get("quiz_target_task_id")
if target_task_id:
    j = local_index_for_task_id(target_task_id)
    st.session_state["quiz_target_task_id"] = None  # czyścimy niezależnie
    if j is not None:
        st.session_state["quiz_pick_local"] = j
    else:
        st.warning("Ten quiz jest już zaliczony albo nie istnieje w quiz.yaml.")

if "quiz_jump_local" not in st.session_state:
    st.session_state["quiz_jump_local"] = None

# jeśli mamy zaplanowany skok, ustaw pick PRZED widgetem
if st.session_state["quiz_jump_local"] is not None:
    st.session_state["quiz_pick_local"] = st.session_state["quiz_jump_local"]
    st.session_state["quiz_jump_local"] = None

# ---- UI: wybór tematu (tylko niezaliczone) ----
pick_local = st.selectbox(
    "Wybierz temat (tylko niezaliczone)",
    range(len(not_done_quiz_indices)),
    format_func=lambda j: not_done_labels[j],
    key="quiz_pick_local",
)

pick = not_done_quiz_indices[pick_local]  # prawdziwy indeks w flat
epic, task = flat[pick]
task_id = task["id"]
task_title = task["title"]
status = store.get_status(task_id)

st.markdown(f"## {epic['id']} — {epic['title']}")
st.markdown(f"### {task_id} — {task_title}")
st.write("**Status:**", status)
st.caption(f"Próg zaliczenia: {int(threshold * 100)}%")

quiz = quizzes.get(task_id)
if quiz is None:
    # teoretycznie nie powinno się zdarzyć, bo filtrujemy po quizzes, ale zostawiamy bezpiecznik
    st.error("Błąd: task jest na liście quizów, ale nie ma go w quizzes dict.")
    st.stop()

st.markdown(f"#### Quiz: {quiz.title}")

# ---- Zbieranie odpowiedzi ----
user_answers = {}
for q in quiz.questions:
    st.markdown(f"**{q.qid}. {q.prompt}**")
    opt_ids = [o["id"] for o in q.choices]
    opt_labels = {o["id"]: o["text"] for o in q.choices}

    choice = st.radio(
        label="",
        options=opt_ids,
        format_func=lambda oid: opt_labels.get(oid, oid),
        index=None,
        key=f"ans_{task_id}_{q.qid}",
    )
    if choice is not None:
        user_answers[q.qid] = choice

    st.divider()

colA, colB = st.columns([1, 1], gap="large")

with colA:
    if st.button("Sprawdź wynik", type="primary", key=f"check_{task_id}"):
        result = grade_single_choice(quiz, user_answers)
        st.session_state["last_result"] = result
        st.session_state["last_task_id"] = task_id

        score = result["score"]
        passed = score >= threshold
        st.session_state["last_passed"] = passed

        if passed:
            store.set_status(task_id, "done")
            nxt = next_local_index(pick_local)
            if nxt is not None:
                st.session_state["quiz_jump_local"] = nxt

                # nie ma kolejnego (albo był ostatni)
                # po rerun lista się zaktualizuje i pokaże "Wszystkie zaliczone ✅"
                pass
            else:
                st.session_state["quiz_pick_local"] = nxt
        else:
            if store.get_status(task_id) == "todo":
                store.set_status(task_id, "in_progress")

        st.rerun()

    # Next (manualny) — tylko po zaliczeniu tego samego taska
    if st.session_state.get("last_task_id") == task_id and st.session_state.get("last_passed") is True:
        nxt = next_local_index(pick_local)
        if nxt is None:
            st.info("Nie ma kolejnego niezaliczonego quizu.")
        else:
            if st.button("➡️ Następny quiz", key=f"next_{task_id}"):
                st.session_state["quiz_pick_local"] = nxt
                st.rerun()

with colB:
    st.markdown("### Feedback")
    last = st.session_state.get("last_result")
    last_task = st.session_state.get("last_task_id")

    if not last or last_task != task_id:
        st.info("Kliknij **Sprawdź wynik**, żeby zobaczyć feedback.")
    else:
        score = last["score"]
        st.metric("Wynik", f"{int(score * 100)}%")
        st.write(f"Poprawne: {last['correct']}/{last['total']}")

        if score >= threshold:
            st.success("Zaliczone ✅")
        else:
            st.error("Nie zaliczone ❌")

        st.divider()
        for d in last["details"]:
            icon = "✅" if d["is_correct"] else "❌"
            st.write(
                f"{icon} **{d['qid']}** — Twoja: `{d['given']}` | Poprawna: `{d['correct']}`"
            )
            if d["explanation"]:
                st.caption(d["explanation"])
