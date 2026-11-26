# Requirements

**Brief:** I need an assistant that answers HR policy questions for shore staff at a shipping company: leave and sick pay, travel and expenses, and working hours, notice periods and the disciplinary process. It should also tell staff how their pension contributions work. Keep a clear, friendly tone.

## Scope
Answers HR policy questions for shore staff at a shipping company regarding leave and sick pay, travel and expenses, working hours, notice periods, and the disciplinary process.

## Audience
shore staff at a shipping company

## Tone
friendly

## Must answer
- leave and sick pay
- travel and expenses
- working hours
- notice periods
- disciplinary process
- pension contributions

## Must refuse
- anything not covered by the provided documents
- pension contributions (not covered by the documents)

## Groundability (can the documents support the brief?)
- **leave and sick pay**: supported (every word of the topic occurs in one sentence of the documents)
- **travel and expenses**: supported
- **working hours**: supported (every word of the topic occurs in one sentence of the documents)
- **notice periods**: supported (every word of the topic occurs in one sentence of the documents)
- **disciplinary process**: partial (model said 'partial', lexical coverage 0.39)
- **pension contributions**: unsupported (no passage shares any words with the topic)

## Clarifying questions
- Q: Are there topics the assistant must refuse or hand off to a human, even if the documents mention them?
  A: (no answer; default assumed)

## Assumptions
- Topic 'pension contributions' taken verbatim from the brief sentence "It should also tell staff how their pension contributions work." (the model's topic list had missed it).
- must_refuse not specified; assumed: anything not covered by the provided documents
