# Requirements

**Brief:** We need an assistant for ship agents calling at Westerhaven Container Terminal. It should answer questions about the terminal regulations, berth booking and berth fees, and certificate validity. It should tell the agent how many days remain until a certificate's expiry date and report the current weather at the port. It should also explain the pilotage fees. It must not give medical advice about crew fitness.

## Scope
The document-based assistant answers questions about Westerhaven Container Terminal regulations, berth booking and fees, certificate validity, pilotage fees, and current weather, without giving medical advice.

## Audience
Ship agents calling at Westerhaven Container Terminal

## Tone
professional, clear and concise

## Must answer
- Terminal Regulations
- Berth Booking
- Berth Fees
- Certificate Validity
- Pilotage Fees
- Current Weather
- many days remain until
- expiry date

## Must refuse
- medical advice about crew fitness
- Pilotage Fees (not covered by the documents)

## Groundability (can the documents support the brief?)
- **Terminal Regulations**: supported (every word of the topic occurs in one sentence of the documents)
- **Berth Booking**: supported (every word of the topic occurs in one sentence of the documents)
- **Berth Fees**: supported (every word of the topic occurs in one sentence of the documents)
- **Certificate Validity**: supported (every word of the topic occurs in one sentence of the documents)
- **Pilotage Fees**: unsupported (model said 'no', lexical coverage 0.28)
- **Current Weather**: supported_by_tool (documents weak; tool 'open_meteo' can serve this)
- **many days remain until**: supported_by_tool (documents weak; tool 'current_date' can serve this)
- **expiry date**: supported (every word of the topic occurs in one sentence of the documents)

## Clarifying questions
- Q: What tone should the assistant use (for example formal, friendly, concise)?
  A: professional, clear and concise
- Q: Does the document 'berth_booking_procedure.md' cover all aspects of berth booking, including priority and changes and cancellations, or are there additional procedures not covered in this document that the assistant should be aware of?
  A: They apply to every vessel calling at the terminal

## Assumptions
- Refusal 'medical advice about crew fitness' taken verbatim from the brief (the model's list had missed it).
- Topic 'many days remain until' taken verbatim from the brief sentence "It should tell the agent how many days remain until a certificate's expiry date and report" (the model's topic list had missed it).
- Topic 'expiry date' taken verbatim from the brief sentence "It should tell the agent how many days remain until a certificate's expiry date and report" (the model's topic list had missed it).
- Clarification: Does the document 'berth_booking_procedure.md' cover all aspects of berth booking, including priority and changes and cancellations, or are there additional procedures not covered in this document that the assistant should be aware of? -> They apply to every vessel calling at the terminal
