from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class QuizQuestion:
    qid: str
    qtype: str
    prompt: str
    choices: List[Dict[str, str]]
    answer: str
    explanation: str


@dataclass(frozen=True)
class Quiz:
    task_id: str
    title: str
    questions: List[QuizQuestion]


def load_quizzes(quiz_yaml: Dict[str, Any]) -> Tuple[float, Dict[str, Quiz]]:
    meta = quiz_yaml.get("meta", {})
    threshold = float(meta.get("pass_threshold", 0.8))

    quizzes_raw = quiz_yaml.get("quizzes", {})
    if not isinstance(quizzes_raw, dict):
        raise ValueError("quiz.yaml: 'quizzes' musi być słownikiem {TASK_ID: {...}}")

    quizzes: Dict[str, Quiz] = {}
    for task_id, qdata in quizzes_raw.items():
        title = qdata.get("title", task_id)
        qs_raw = qdata.get("questions", [])
        if not isinstance(qs_raw, list):
            raise ValueError(f"quiz.yaml: questions dla {task_id} musi być listą")

        questions: List[QuizQuestion] = []
        for item in qs_raw:
            q = QuizQuestion(
                qid=str(item.get("id")),
                qtype=str(item.get("type", "single_choice")),
                prompt=str(item.get("prompt", "")),
                choices=list(item.get("choices", [])),
                answer=str(item.get("answer", "")),
                explanation=str(item.get("explanation", "")),
            )
            if q.qtype != "single_choice":
                raise ValueError(f"Na razie wspieram tylko single_choice. Problem w {task_id}/{q.qid}")
            questions.append(q)

        quizzes[task_id] = Quiz(task_id=task_id, title=title, questions=questions)

    return threshold, quizzes


def grade_single_choice(quiz: Quiz, user_answers: Dict[str, str]) -> Dict[str, Any]:
    total = len(quiz.questions)
    correct = 0
    details = []

    for q in quiz.questions:
        given = user_answers.get(q.qid)
        is_correct = (given == q.answer)
        if is_correct:
            correct += 1
        details.append(
            {
                "qid": q.qid,
                "given": given,
                "correct": q.answer,
                "is_correct": is_correct,
                "explanation": q.explanation,
            }
        )

    score = (correct / total) if total > 0 else 0.0
    return {"total": total, "correct": correct, "score": score, "details": details}
