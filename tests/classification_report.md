# Query Classifier Test Report

**Model**: openai/qwen3

**Total Questions**: 40
**Passed**: 35
**Failed**: 5
**Accuracy**: 87.5%

## Positive Tests (should classify as 'receipt')

| Question | Expected | Actual | Status |
|----------|----------|--------|--------|
| How much did I buy Widget X for? | receipt | receipt | PASS |
| What was my average spending for Acme Energy? | receipt | statement | FAIL |
| Has my Contoso bill ever gone above $100 and if so when? | receipt | bill | FAIL |
| How much did I pay for my last electricity bill? | receipt | bill | FAIL |
| What did I spend at the supermarket last week? | receipt | receipt | PASS |
| How much was my medical consultation? | receipt | receipt | PASS |
| What was the total cost of my online order from Amazon? | receipt | receipt | PASS |
| How much did I pay for my coffee this morning? | receipt | receipt | PASS |
| What was the price of my train ticket? | receipt | receipt | PASS |
| How much did I spend on groceries last month? | receipt | statement | FAIL |
| What did I pay for my gym membership? | receipt | receipt | PASS |
| How much was my dental checkup? | receipt | receipt | PASS |
| What was the total at the petrol station? | receipt | receipt | PASS |
| How much did I pay for my internet bill? | receipt | receipt | PASS |
| What did I spend on my lunch? | receipt | receipt | PASS |
| How much was my phone bill? | receipt | bill | FAIL |
| What was the cost of my Uber ride? | receipt | receipt | PASS |
| How much did I pay for my Netflix subscription? | receipt | receipt | PASS |
| What was the price of my prescription? | receipt | receipt | PASS |
| How much did I spend on my flight ticket? | receipt | receipt | PASS |

## Negative Tests (should NOT classify as 'receipt')

| Question | Expected | Actual | Status |
|----------|----------|--------|--------|
| What was my bank balance at the end of last month? | not_receipt | statement | PASS |
| Show me my credit card statement for January | not_receipt | statement | PASS |
| What transactions did I make in February? | not_receipt | statement | PASS |
| Did I receive my payslip for this month? | not_receipt | payslip | PASS |
| What is my current account balance? | not_receipt | statement | PASS |
| When is my policy renewal date? | not_receipt | policy | PASS |
| What are the terms and conditions of my contract? | not_receipt | terms and conditions | PASS |
| Can I see my enrollment details for the course? | not_receipt | enrolment | PASS |
| What was my tax declaration amount? | not_receipt | declaration | PASS |
| When does my warranty expire? | not_receipt | warranty | PASS |
| What are the specs for the laptop I ordered? | not_receipt | specification | PASS |
| When is my flight booking? | not_receipt | booking | PASS |
| What is my employee share scheme allocation? | not_receipt | employee share scheme | PASS |
| When is my visa expiration date? | not_receipt | visa | PASS |
| What are the booking details for my hotel? | not_receipt | booking | PASS |
| What was my tax invoice number? | not_receipt | invoice | PASS |
| Can I see my prescription details? | not_receipt | prescription | PASS |
| When is my next appointment scheduled? | not_receipt | booking | PASS |
| What are the lodgement requirements? | not_receipt | lodgement | PASS |
| What was my inventory count for last quarter? | not_receipt | inventory | PASS |
