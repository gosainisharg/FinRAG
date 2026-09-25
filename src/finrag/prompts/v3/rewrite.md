---
name: rewrite
version: v3
description: Reformulates a query after retrieval came back insufficient
variables: question, missing, previous_queries
---
A search over SEC filings did not find enough to answer the question. Write a better search
query.

What was missing: {missing}

Queries already tried (do not repeat them):
{previous_queries}

Guidance:
- Use the vocabulary a filing would use, not the vocabulary of the question. Filings say
  "total net sales" where a person says "revenue", "provision for income taxes" where a
  person says "tax bill", "property, plant and equipment" where a person says "factories".
- Add the specific line-item name if you can infer it.
- Keep the company and period in the query.
- Write a search query, not a sentence. No question mark, no politeness.

ORIGINAL QUESTION: {question}
