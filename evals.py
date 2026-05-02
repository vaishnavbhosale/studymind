def keyword_score(answer, expected_keywords):
    """Check if answer contains important keywords"""
    if not expected_keywords:
        return 0.0

    score = 0
    for keyword in expected_keywords:
        if keyword.lower() in answer.lower():
            score += 1

    return score / len(expected_keywords)


def llm_judge(client, question, answer, context):
    """Use AI to judge if answer is correct"""
    prompt = f"""
    Question: {question}
    Context: {context}
    Answer: {answer}

    Is the answer correct AND based only on the context?
    Reply only with YES or NO.
    """

    result = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return "YES" in result.text.upper()