"""
evals.py

Evaluation utilities: keyword scoring and LLM-as-judge.
Kept as a standalone module (not inside services/ or agent.py)
so it can be imported by rag_service.py without creating any
import cycle.
"""


def keyword_score(answer: str, expected_keywords: list[str]):
    """
    Returns the fraction of expected_keywords present in the answer.

    Returns None (not 0.0) when no keywords are provided,
    so callers can distinguish 'nothing to check' from 'check failed'.
    The agent treats None as 'not applicable' and relies solely on
    the LLM judge in that case.
    """
    if not expected_keywords:
        return None
    score = sum(1 for kw in expected_keywords if kw.lower() in answer.lower())
    return score / len(expected_keywords)


def llm_judge(client, question: str, answer: str, context: str) -> bool:
    """
    Scores the answer on three dimensions and returns True (PASS)
    only if all three scores exceed 0.7.

    Upgraded from YES/NO to three rubric scores so failures are
    diagnosable: a low FAITHFULNESS score means hallucination,
    a low COMPLETENESS score means the answer is too shallow.

    Parsing targets the explicit VERDICT line to avoid false
    positives from words like COMPASS or BYPASS that contain
    'PASS' as a substring.
    """
    prompt = f"""
    Question: {question}
    Context: {context}
    Answer: {answer}

    Score each from 0.0 to 1.0:
    FAITHFULNESS: Is every claim in the answer supported by context?
    RELEVANCE: Does the answer address the question?
    COMPLETENESS: Are the key points covered?

    Reply EXACTLY like this (no extra lines):
    FAITHFULNESS: <score>
    RELEVANCE: <score>
    COMPLETENESS: <score>
    VERDICT: <PASS if all above 0.7 else FAIL>
    """

    result = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )

    for line in result.text.upper().splitlines():
        line = line.strip()
        if line.startswith("VERDICT:"):
            return "PASS" in line.split(":", 1)[1]

    # If model didn't follow the format, default to failing safely
    return False


def hallucination_guard(answer: str) -> bool:
    """
    Returns True if the answer is a valid refusal for an out-of-scope question.
    Used in evals to confirm the system correctly declines rather than hallucinating.
    """
    refusal_phrases = [
        "couldn't find this in your notes",
        "not in your notes",
        "cannot find",
        "not available in your notes"
    ]
    return any(phrase in answer.lower() for phrase in refusal_phrases)