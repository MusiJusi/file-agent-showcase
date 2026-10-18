# Try it with sample documents

These documents are fictional and contain no user data.
Upload both [the quarterly report](quarterly_report.md) and
[the service agreement](service_agreement.md) through the Streamlit sidebar.

| Ask | Expected facts to verify | What it exercises |
| --- | --- | --- |
| What was North's revenue growth from Q1 to Q2? | 25%, from 100 to 125 | Table reading and arithmetic |
| Which regions exceeded their Q2 target? | North: 125 vs 120; West: 132 vs 130 | Complete table comparison |
| What service credit applies to the reported Q2 availability? | 5% for 99.3% | Evidence from both documents |
| When must a claim be submitted? | Within 15 calendar days after month end | Direct source lookup |
| What is the service account number? | Not specified | Grounded abstention |

These are expected facts, not recorded LLM outputs.
Inspect the returned sources as well as the answer.
Wording and tool choices depend on the model.

Markdown tables use the shared block representation also used by parsed
spreadsheets. Real XLSX files additionally preserve sheet metadata.
