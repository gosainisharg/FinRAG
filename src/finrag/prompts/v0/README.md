# v0 — controls, not production

These are deliberately naive prompts, kept so the engineered versions can be measured
against something rather than merely asserted to be better.

`grade.md` is the prompt used in the Phase 0 experiment. Measured on five grader cases,
three trials each, over this corpus:

| model | v0 (naive) | v1 (engineered) |
|---|:--:|:--:|
| llama3.2:3b | 9/15 | 12/15 |
| mistral:7b | 6/15 | **15/15** |
| llama3.1:8b | 6/15 | **15/15** |

Schema compliance was 15/15 everywhere — structured output was never the problem. The
failure was that every model answered `sufficient: true` for questions the context could
not answer, which leaves the corrective loop silently never firing. Note also that the
naive prompt made the *larger* models score worse than the smaller one.

Do not set `FINRAG_PROMPT_VERSION=v0`. Use `FINRAG_PROMPT_VERSIONS='{"grade": "v0"}'` to
swap this one prompt for a comparison run.
