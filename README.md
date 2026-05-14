# Cedar Kids Therapy — Referral Inbox Triage Agent

## How to Run

```bash
npm install
export ANTHROPIC_API_KEY=your_key_here
npm run triage   -- --input data/inbox.json --output output.json --trace .trace/tool-calls.jsonl
npm run validate -- --input data/inbox.json --output output.json --trace .trace/tool-calls.jsonl
```

Both commands default to the paths above if no flags are provided.

---

## Stack

- **TypeScript**, Node LTS, ESM
- **Anthropic Claude Haiku** via `@anthropic-ai/sdk` — two LLM calls per item (classify + draft)
- **LangGraph** — state graph with typed nodes per routing tier
- **AJV** — output validated against `schema/output.schema.json`

---

## Architecture

### Two-Level Classification

Every item is assigned a coarse tier before a specific classification:

```
Tier (coarse)     Classification (specific)
risk           →  safeguarding
intake         →  new_referral, missing_paperwork
patient_ops    →  scheduling, existing_patient_request, billing_question
clinical       →  clinical_question, provider_followup
dead_end       →  spam, other, complaint
```

The tier gates routing. A safeguarding item never reaches the intake branch.

### LangGraph State Graph

```
[triage] — LLM classify + keyword safeguarding scan
    │
    ├── risk         → escalate + task + internal draft
    ├── intake       → verify_insurance → find_slots → hold_slot? → draft + task
    ├── patient_ops  → find_slots + task + draft
    ├── clinical     → lookup_policy + draft + task
    └── dead_end     → no tools
```

### Two LLM Calls Per Item

1. **Classify** — structured prompt returns all 7 intake fields + classification + urgency + rationale as JSON in one shot
2. **Draft** — context-aware prompt per branch handles Spanish detection, internal-only flag for safeguarding, missing fields list for incomplete referrals, and non-clinical constraints for clinical questions

### Key Decision Points

**Safeguarding gate** — a keyword scan runs on every item before the LLM result is accepted. If the scan fires and the LLM missed it, classification is overridden to `safeguarding` and urgency forced to `P0`. The LLM is the primary classifier; the scan is the safety net.

**hold_slot** — only fires for fax referrals that are in-network with no missing fields. Prevents slots being held for families who cannot confirm.

**requires_human_review** — computed per item based on classification, urgency, missing fields, and escalation, the logic supports `false` for fully resolved clean referrals.

---

## Tradeoffs and Cuts

**Sequential item processing.** Items run one at a time. A production agent would use `Promise.allSettled` with per-item error isolation so one failure doesn't block the batch.

**Slot preferences not passed to find_slots.** The agent finds an open appointment slot but doesn't check if it actually fits the family.


## With More Time

- Preference-aware slot matching using extracted parent availability
- Parallel item processing with per-item error isolation
- Confidence threshold on LLM output — low-confidence items auto-flagged for review
- Labeled eval set for safeguarding edge cases and regression testing before any prompt change
