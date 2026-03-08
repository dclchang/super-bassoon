import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src" / "super-bassoon"))

import asyncio
import json
from super_bassoon.llmproxy import LlmProxy
from super_bassoon.op import get_secret


DOCUMENT_TYPES = [
    'application', 'article', 'assessment', 'assignment', 'authorisation', 'bill', 'book',
    'booking', 'booklist', 'brochure', 'calendar', 'catalog', 'certificate', 'checklist',
    'claim', 'contactlist', 'contract', 'declaration', 'document', 'employee share scheme',
    'enrolment', 'factsheet', 'fine', 'guide', 'handbook', 'id', 'image', 'inventory',
    'invoice', 'itemlist', 'itinerary', 'letter', 'listing', 'lodgement', 'log', 'manual',
    'map', 'newsletter', 'pamphlet', 'passport', 'payslip', 'photo', 'plans', 'policy',
    'power of attorney', 'premium', 'prescription', 'pricelist', 'prospectus', 'quotation',
    'receipt', 'recipe', 'registration', 'report', 'resume', 'review', 'schedule', 'score',
    'specification', 'statement', 'statutory declaration', 'submission', 'syllabus',
    'terms and conditions', 'ticket', 'timetable', 'transcript', 'visa', 'warranty',
    'workbook', 'ytd'
]

POSITIVE_QUESTIONS = [
    "How much did I buy Widget X for?",
    "What was my average spending for Acme Energy?",
    "Has my Contoso bill ever gone above $100 and if so when?",
    "How much did I pay for my last electricity bill?",
    "What did I spend at the supermarket last week?",
    "How much was my medical consultation?",
    "What was the total cost of my online order from Amazon?",
    "How much did I pay for my coffee this morning?",
    "What was the price of my train ticket?",
    "How much did I spend on groceries last month?",
    "What did I pay for my gym membership?",
    "How much was my dental checkup?",
    "What was the total at the petrol station?",
    "How much did I pay for my internet bill?",
    "What did I spend on my lunch?",
    "How much was my phone bill?",
    "What was the cost of my Uber ride?",
    "How much did I pay for my Netflix subscription?",
    "What was the price of my prescription?",
    "How much did I spend on my flight ticket?"
]

NEGATIVE_QUESTIONS = [
    "What was my bank balance at the end of last month?",
    "Show me my credit card statement for January",
    "What transactions did I make in February?",
    "Did I receive my payslip for this month?",
    "What is my current account balance?",
    "When is my policy renewal date?",
    "What are the terms and conditions of my contract?",
    "Can I see my enrollment details for the course?",
    "What was my tax declaration amount?",
    "When does my warranty expire?",
    "What are the specs for the laptop I ordered?",
    "When is my flight booking?",
    "What is my employee share scheme allocation?",
    "When is my visa expiration date?",
    "What are the booking details for my hotel?",
    "What was my tax invoice number?",
    "Can I see my prescription details?",
    "When is my next appointment scheduled?",
    "What are the lodgement requirements?",
    "What was my inventory count for last quarter?"
]


def load_prompt() -> str:
    prompt_path = Path(__file__).parents[1] / "src" / "super-bassoon" / "prompts" / "query" / "classify" / "receipt.txt"
    prompt = prompt_path.read_text()
    return prompt.replace("{document_types}", ", ".join(DOCUMENT_TYPES))


async def main():
    llm = LlmProxy(
        base_url="http://192.168.68.222:4040",
        api_key=get_secret("op://homelab/litellm-virtual-key-for-rag-app/credential"),
        models={
            "extractor": "openai/qwen3",
            "reviewer": "openai/qwen3",
            "embedding": "openai/nomic_embed_text"
        }
    )

    system_msg = load_prompt()

    results = []

    for question in POSITIVE_QUESTIONS:
        result = await llm.chat(
            model="openai/qwen3",
            prompt=question,
            system=system_msg,
            is_json=False
        )
        classification = result.strip().lower()
        expected = "receipt"
        passed = classification == expected
        results.append({
            "question": question,
            "expected": expected,
            "actual": classification,
            "passed": passed
        })

    for question in NEGATIVE_QUESTIONS:
        result = await llm.chat(
            model="openai/qwen3",
            prompt=question,
            system=system_msg,
            is_json=False
        )
        classification = result.strip().lower()
        expected = "not_receipt"
        passed = classification != "receipt"
        results.append({
            "question": question,
            "expected": expected,
            "actual": classification,
            "passed": passed
        })

    passed_count = sum(1 for r in results if r["passed"])
    total_count = len(results)
    accuracy = (passed_count / total_count) * 100

    report_path = Path(__file__).parent / "classification_report.md"
    with open(report_path, "w") as f:
        f.write("# Query Classifier Test Report\n\n")
        f.write(f"**Model**: openai/qwen3\n\n")
        f.write(f"**Total Questions**: {total_count}\n")
        f.write(f"**Passed**: {passed_count}\n")
        f.write(f"**Failed**: {total_count - passed_count}\n")
        f.write(f"**Accuracy**: {accuracy:.1f}%\n\n")
        f.write("## Positive Tests (should classify as 'receipt')\n\n")
        f.write("| Question | Expected | Actual | Status |\n")
        f.write("|----------|----------|--------|--------|\n")
        for r in results[:20]:
            status = "PASS" if r["passed"] else "FAIL"
            f.write(f"| {r['question']} | {r['expected']} | {r['actual']} | {status} |\n")

        f.write("\n## Negative Tests (should NOT classify as 'receipt')\n\n")
        f.write("| Question | Expected | Actual | Status |\n")
        f.write("|----------|----------|--------|--------|\n")
        for r in results[20:]:
            status = "PASS" if r["passed"] else "FAIL"
            f.write(f"| {r['question']} | {r['expected']} | {r['actual']} | {status} |\n")

    print(f"Report written to {report_path}")
    print(f"\nResults: {passed_count}/{total_count} passed ({accuracy:.1f}%)")


if __name__ == "__main__":
    asyncio.run(main())
