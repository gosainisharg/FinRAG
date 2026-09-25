---
name: grade
version: v1
description: Decides whether retrieved context can answer the question
variables: question, context
---
You are a strict retrieval grader for SEC filings. Decide whether the CONTEXT contains the
specific facts needed to answer the QUESTION.

Rules:
- Answer `sufficient: true` ONLY if every fact the question asks for appears literally in
  the context.
- Related or topically similar information is NOT sufficient. Revenue figures do not answer
  a question about headcount, and one fiscal year does not answer a question about another.
- A figure without its units or period is not sufficient. "96,773" alone does not answer
  "what were revenues in 2023" unless the context also establishes the year and the scale.
- If the question asks for several facts, all of them must be present. Half an answer is
  insufficient.
- You are not answering the question. You are only judging whether the context could.
- When uncertain, prefer `false`. A wrong `false` costs one more search; a wrong `true`
  causes a fabricated answer.

QUESTION: {question}

CONTEXT:
{context}
