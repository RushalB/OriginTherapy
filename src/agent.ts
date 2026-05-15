import type {
  InboxItem,
  ItemOutput,
  Classification,
  ClassificationTier,
  ExtractedIntake,
  Urgency,
  Assignee,
  Channel,
} from "./types.js";
import {
  search_patient,
  verify_insurance,
  lookup_policy,
  find_slots,
  hold_slot,
  create_task,
  draft_message,
  escalate,
  withItemContext,
  getToolCallsForItem,
} from "./tools.js";

const SPANISH_RE = /evaluacion|hola|gracias|para.*evaluaci/i;

// ── LOOKUP TABLES ────────────────────────────────────────────────────────────

/** Maps Inbox channel types to draft message channel types */
function mapChannel(channel: Channel): "portal" | "email" | "phone" {
  if (channel === "fax_referral") return "email";
  if (channel === "portal_message") return "portal";
  if (channel === "voicemail_transcript") return "phone";
  return "email";
}

/** Maps each Classification to its routing tier */
export const TIER_MAP: Record<Classification, ClassificationTier> = {
  safeguarding:              "risk",
  new_referral:              "intake",
  missing_paperwork:         "intake",
  scheduling:                "patient_ops",
  existing_patient_request:  "patient_ops",
  billing_question:          "patient_ops",
  clinical_question:         "clinical",
  provider_followup:         "dead_end",
  complaint:                 "dead_end",
  spam:                      "dead_end",
  other:                     "dead_end",
};

/** Minimum urgency floor per tier — agent must not assign below this */
export const URGENCY_FLOOR: Record<ClassificationTier, Urgency> = {
  risk:        "P0",
  intake:      "P2",
  patient_ops: "P2",
  clinical:    "P3",
  dead_end:    "P3",
};

/** Default staff assignee per tier for create_task */
export const DEFAULT_ASSIGNEE: Record<ClassificationTier, Assignee> = {
  risk:        "clinical_lead",
  intake:      "intake",
  patient_ops: "front_desk",
  clinical:    "clinical_lead",
  dead_end:    "front_desk",
};


// ── LLM CLASSIFICATION LAYER ──────────────────────────────────────────────────

/** Quick keyword scan for safeguarding signals */
export function hasSafeguardingSignal(body: string): boolean {
  const keywords = [/rough with/i, /getting rough/i, /hit/i, /abuse/i, /neglect/i, /harm/i, /unsafe/i];
  return keywords.some(re => re.test(body));
}

export interface ClassifyResult {
  extracted_intake: ExtractedIntake;
  classification: Classification;
  urgency: Urgency;
  missing_info: string[];
  decision_rationale: string;
}

/** Determines if an item requires human oversight based on clinic safety policies */
export function requiresHumanReview(
  classification: Classification,
  urgency: Urgency,
  missing_info: string[],
  escalation: BranchResult["escalation"],
): boolean {
  // Always escalated items
  if (escalation !== null) return true;
  if (classification === "safeguarding") return true;

  // New referrals always require intake review to finalize scheduling
  if (classification === "new_referral" || classification === "missing_paperwork") return true;

  // Same-day operational issues
  if (urgency === "P0" || urgency === "P1") return true;

  // Incomplete data — human must unblock
  if (missing_info.length > 0) return true;

  // Clinical questions — can't auto-resolve
  if (classification === "clinical_question") return true;

  return false;
}

/** Builds the structured prompt for generating an empathetic draft reply */
export function buildDraftPrompt(
  item: InboxItem,
  extracted: ExtractedIntake,
  classification: Classification,
  context: Record<string, unknown>,
): string {
  return `You are a care coordinator at Cedar Kids Therapy, a pediatric therapy clinic.
Write a draft reply for the following inbox item.

Constraints:
- Empathetic, warm, concise (3-5 sentences max)
- No clinical advice, no diagnosis, no developmental predictions
- Do NOT imply the message has already been sent
- Do NOT use phrases like "I have sent" or "you will receive" — this is a draft only
${item.channel === "voicemail_transcript" && SPANISH_RE.test(item.body) ? "- Write the reply in Spanish" : ""}
${classification === "safeguarding" ? "- INTERNAL STAFF USE ONLY. Do not address the family. Neutral language only." : ""}
${classification === "clinical_question" ? "- Do not answer the clinical question. Offer a screening appointment as an option, not a recommendation." : ""}

Item:
Channel: ${item.channel}
From: ${item.sender}
Subject: ${item.subject}
Body: ${item.body}

Return ONLY the draft reply text. No subject line, no sign-off placeholder, no explanation.`;
}

/** Calls the LLM to generate the draft reply */
export async function callDraftLLM(prompt: string): Promise<string> {
  try {
    const { default: Anthropic } = await import("@anthropic-ai/sdk");
    const client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
    const response = await client.messages.create({
      model: "claude-haiku-4-5-20251001",
      max_tokens: 256,
      messages: [{ role: "user", content: prompt }],
    });
    return response.content
      .filter((b) => b.type === "text")
      .map((b) => (b as { type: "text"; text: string }).text)
      .join("")
      .trim();
  } catch (error) {
    console.warn("Draft LLM failed, using fallback:", error);
    return "Thank you for reaching out. A member of our team will follow up with you shortly to assist you with your request.";
  }
}

/** Builds the structured prompt asking the LLM for all fields in one shot */
export function buildClassifyPrompt(item: InboxItem): string {
  return `You are a triage assistant for Cedar Kids Therapy, a pediatric therapy clinic (SLP, OT, PT).
Analyze the following inbox message and return a JSON object with EXACTLY these fields:

{
  "child_name": string | null,
  "dob_or_age": string | null,          // ISO date "YYYY-MM-DD" or age like "6"
  "parent_contact": string | null,       // name, phone, email concatenated
  "discipline": ["SLP"|"OT"|"PT"] | null,
  "diagnosis_or_concern": string | null,
  "payer": string | null,
  "member_id": string | null,
  "classification": one of: "new_referral"|"existing_patient_request"|"scheduling"|
    "clinical_question"|"billing_question"|"missing_paperwork"|"provider_followup"|
    "complaint"|"safeguarding"|"spam"|"other",
  "urgency": "P0"|"P1"|"P2"|"P3",
  "missing_info": string[],              // field names absent that are required for this type
  "decision_rationale": string           // 1-2 sentences explaining classification
}

Urgency rules:
- P0: safeguarding, immediate harm, today's appointment at risk
- P1: same-day cancellation/reschedule, incomplete referral needing urgent follow-up
- P2: routine new referral (default)
- P3: general question, no booking intent

IMPORTANT: Return ONLY the JSON object. No markdown, no explanation, no code fences.

--- INBOX MESSAGE ---
Channel: ${item.channel}
Subject: ${item.subject}
From: ${item.sender}
Body:
${item.body}`;
}

const VALID_CLASSIFICATIONS = new Set<string>([
  "new_referral", "existing_patient_request", "scheduling", "clinical_question",
  "billing_question", "missing_paperwork", "provider_followup", "complaint",
  "safeguarding", "spam", "other",
]);
const VALID_URGENCIES = new Set<string>(["P0", "P1", "P2", "P3"]);

interface RawLLMOutput {
  child_name: unknown;
  dob_or_age: unknown;
  parent_contact: unknown;
  discipline: unknown;
  diagnosis_or_concern: unknown;
  payer: unknown;
  member_id: unknown;
  classification: unknown;
  urgency: unknown;
  missing_info: unknown;
  decision_rationale: unknown;
}

/** Calls the Anthropic API and validates the shape of the response */
export async function callLLM(prompt: string): Promise<ClassifyResult> {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) throw new Error("ANTHROPIC_API_KEY not set");

  const { default: Anthropic } = await import("@anthropic-ai/sdk");
  const client = new Anthropic({ apiKey });

  const response = await client.messages.create({
    model: "claude-haiku-4-5-20251001",
    max_tokens: 1024,
    messages: [{ role: "user", content: prompt }],
  });

  const text = response.content
    .filter((b) => b.type === "text")
    .map((b) => (b as { type: "text"; text: string }).text)
    .join("");

  let cleanText = text.trim();
  if (cleanText.startsWith("```")) {
    cleanText = cleanText.replace(/^```(?:json)?\n?/, "").replace(/\n?```$/, "");
  }

  let raw: RawLLMOutput;
  try {
    raw = JSON.parse(cleanText) as RawLLMOutput;
  } catch {
    throw new Error(`LLM returned non-JSON: ${cleanText.slice(0, 200)}`);
  }

  // Validate required enum fields
  if (!VALID_CLASSIFICATIONS.has(String(raw.classification))) {
    throw new Error(`Invalid classification: ${raw.classification}`);
  }
  if (!VALID_URGENCIES.has(String(raw.urgency))) {
    throw new Error(`Invalid urgency: ${raw.urgency}`);
  }

  const nullableString = (v: unknown): string | null =>
    typeof v === "string" && v.trim().length > 0 ? v.trim() : null;

  const discipline = Array.isArray(raw.discipline)
    ? (raw.discipline as string[]).filter((d): d is "SLP" | "OT" | "PT" =>
        ["SLP", "OT", "PT"].includes(d),
      )
    : null;

  return {
    extracted_intake: {
      child_name: nullableString(raw.child_name),
      dob_or_age: nullableString(raw.dob_or_age),
      parent_contact: nullableString(raw.parent_contact),
      discipline: discipline && discipline.length > 0 ? discipline : null,
      diagnosis_or_concern: nullableString(raw.diagnosis_or_concern),
      payer: nullableString(raw.payer),
      member_id: nullableString(raw.member_id),
    },
    classification: raw.classification as Classification,
    urgency: raw.urgency as Urgency,
    missing_info: Array.isArray(raw.missing_info)
      ? (raw.missing_info as unknown[]).map(String)
      : [],
    decision_rationale: nullableString(raw.decision_rationale) ?? "",
  };
}

/**
 * Primary classification entry point.
 * Calls the Anthropic LLM to extract data and classify the item.
 * Falls back to a safe "human review" state if the LLM fails.
 */
export async function classifyWithLLM(item: InboxItem): Promise<ClassifyResult> {
  try {
    const prompt = buildClassifyPrompt(item);
    return await callLLM(prompt);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error(`[classifyWithLLM] CRITICAL FAILURE for ${item.id}: ${msg}`);
    return {
      extracted_intake: {
        child_name: null,
        dob_or_age: null,
        parent_contact: null,
        discipline: null,
        diagnosis_or_concern: null,
        payer: null,
        member_id: null,
      },
      classification: "other",
      urgency: "P1",
      missing_info: [],
      decision_rationale: "LLM classification failed. Routing to human review for safety.",
    };
  }
}

// ── HELPER: isCleanReferral ──────────────────────────────────────────────────

/**
 * Clean referral definition:
 * channel === fax_referral AND in_network AND no missing critical fields
 */
export function isCleanReferral(
  item: InboxItem,
  extracted: ExtractedIntake,
  insuranceStatus: string
): boolean {
  if (item.channel !== "fax_referral") return false;
  if (insuranceStatus !== "in_network") return false;
  if (!extracted.child_name || !extracted.dob_or_age || !extracted.payer || !extracted.discipline || extracted.discipline.length === 0) return false;
  return true;
}

// ── BRANCH: RISK / SAFEGUARDING ──────────────────────────────────────────────

export interface BranchResult {
  missing_info: string[];
  recommended_next_action: string;
  draft_reply: string | null;
  task_ids: string[];
  escalation: { reason: string; severity: "P0" | "P1" } | null;
  decision_rationale: string;
}

export async function runSafeguardingBranch(
  item: InboxItem,
  extracted: ExtractedIntake,
  classification: Classification,
  context: Record<string, unknown>,
): Promise<BranchResult> {
  // 1. Pull safeguarding policy so the rationale is grounded
  await lookup_policy({ topic: "safeguarding" });

  // 2. Escalate immediately — P0, mandatory reporter review
  const esc = await escalate({
    item_id: item.id,
    reason:
      "Message contains disclosure suggesting potential physical harm or unsafe caregiving. " +
      "Mandatory reporter review required.",
    severity: "P0",
  });

  // 3. Create a same-hour review task for the clinical lead
  const task = await create_task({
    assignee: "clinical_lead",
    title: `Mandatory reporter review — ${extracted.child_name ?? "child"} (${item.id})`,
    due: new Date().toISOString(),
    notes:
      "Safeguarding signal detected in inbound message. Do not contact the family until " +
      "mandatory reporting obligations have been assessed. Review immediately.",
  });

  // 4. Draft internal-only message — neutral, no investigative advice per policy
  const draftBody = await callDraftLLM(
    buildDraftPrompt(item, extracted, classification, context)
  );

  const msg = await draft_message({
    recipient: "clinical_lead",
    channel: "email",
    body: draftBody,
  });

  return {
    missing_info: [],
    recommended_next_action:
      "Immediate clinical lead review — assess mandatory reporting obligations before any family contact.",
    draft_reply: msg.data.draft_id,
    task_ids: [task.data.task_id],
    escalation: {
      reason: esc.args["reason"] as string,
      severity: "P0",
    },
    decision_rationale:
      "Message contained language suggesting potential physical harm to a child. " +
      "Per safeguarding policy, this is P0: escalated to clinical lead immediately with a " +
      "same-hour review task. No outbound message sent to family.",
  };
}

// ── BRANCH: INTAKE ─────────────────────────────────────────────────────────────

export async function runIntakeBranch(
  item: InboxItem,
  extracted: ExtractedIntake,
  patientResult: { patient_id: string } | null,
  classification: Classification,
  context: Record<string, unknown>,
): Promise<BranchResult> {
  const missing_info: string[] = [];
  const task_ids: string[] = [];

  // 1. Language policy if Spanish
  if (context.is_spanish) {
    await lookup_policy({ topic: "language_access" });
  }

  // 2. Verify Insurance
  const ins = await verify_insurance({
    payer: extracted.payer ?? undefined,
    member_id: extracted.member_id ?? undefined,
  });

  // Out-of-network early return
  if (ins.data.status === "out_of_network") {
    missing_info.push("Insurance out-of-network — benefits review required before scheduling");
    await lookup_policy({ topic: "insurance" });
    const task = await create_task({
      assignee: "billing",
      title: `OON Benefits Check — ${extracted.child_name ?? "Unknown"}`,
      due: new Date().toISOString(),
      notes: "Referral is out-of-network. Benefits conversation required.",
    });
    
    const draftBody = await callDraftLLM(
      buildDraftPrompt(item, extracted, classification, context)
    );

    const msg = await draft_message({
      recipient: "parent",
      channel: context.is_spanish ? "phone" : "email",
      body: draftBody,
    });

    return {
      missing_info,
      recommended_next_action: "Billing team needs to conduct an OON benefits conversation.",
      draft_reply: msg.data.draft_id,
      task_ids: [task.data.task_id],
      escalation: null,
      decision_rationale: "Insurance verified as out-of-network. Created a billing task per policy and drafted a notification message to the parent.",
    };
  }

  // 3. Find slots
  const slots = await find_slots({
    discipline: extracted.discipline?.[0] ?? "SLP",
  });

  // 4. Hold slot if clean
  const isClean = isCleanReferral(item, extracted, ins.data.status);
  let rationale = `Insurance verified as ${ins.data.status}. `;

  if (isClean && slots.data.length > 0) {
    await hold_slot({
      slot_id: slots.data[0].slot_id,
      patient_ref: patientResult?.patient_id ?? "unknown_patient",
    });
    rationale += `Clean referral detected; held a provisional slot. `;
  } else {
    rationale += `Did not hold slot (clean=${isClean}). `;
  }

  // 5. Draft message & Create intake task
  const draftBody = await callDraftLLM(
    buildDraftPrompt(item, extracted, classification, context)
  );

  const msg = await draft_message({
    recipient: "parent",
    channel: context.is_spanish ? "phone" : "email",
    body: draftBody,
  });

  const task = await create_task({
    assignee: "intake",
    title: `Process New Referral — ${extracted.child_name ?? "Unknown"}`,
    due: new Date().toISOString(),
    notes: `Verify demographics, confirm benefits, and finalize scheduling.`,
  });

  return {
    missing_info,
    recommended_next_action: "Intake team needs to complete registration and finalize scheduling.",
    draft_reply: msg.data.draft_id,
    task_ids: [task.data.task_id],
    escalation: null,
    decision_rationale: rationale + "Drafted parent notification and created intake task.",
  };
}

// ── BRANCH: PATIENT OPS ───────────────────────────────────────────────────────

export async function runPatientOpsBranch(
  item: InboxItem,
  extracted: ExtractedIntake,
  classification: Classification,
  context: Record<string, unknown>,
): Promise<BranchResult> {
  // 1. Policy check for scheduling/cancellation
  await lookup_policy({ topic: "scheduling" });

  // 2. Find slots if they are asking for reschedule
  const discipline = extracted.discipline?.[0] || "OT";
  await find_slots({ discipline });

  // 3. Create task for front desk (due today)
  const task = await create_task({
    assignee: "front_desk",
    title: `Patient Ops — ${extracted.child_name ?? "Patient"}`,
    due: new Date().toISOString(),
    notes: `Patient requested a reschedule, billing update, or has an operational question. Please review and contact them.`,
  });

  // 4. Draft message
  const draftBody = await callDraftLLM(
    buildDraftPrompt(item, extracted, classification, context)
  );

  const msg = await draft_message({
    recipient: item.sender,
    channel: mapChannel(item.channel),
    body: draftBody,
  });

  return {
    missing_info: [],
    recommended_next_action: "Front desk review for scheduling/billing/ops.",
    draft_reply: msg.data.draft_id,
    task_ids: [task.data.task_id],
    escalation: null,
    decision_rationale: "Patient request regarding scheduling or operations. Routed to front desk with a same-day task.",
  };
}

// ── BRANCH: CLINICAL ─────────────────────────────────────────────────────────

export async function runClinicalBranch(
  item: InboxItem,
  extracted: ExtractedIntake,
  classification: Classification,
  context: Record<string, unknown>,
): Promise<BranchResult> {
  // 1. Policy check
  await lookup_policy({ topic: "clinical_advice" });

  // 2. Draft warm, non-clinical response
  const draftBody = await callDraftLLM(
    buildDraftPrompt(item, extracted, classification, context)
  );

  const msg = await draft_message({
    recipient: item.sender,
    channel: mapChannel(item.channel),
    body: draftBody,
  });

  // 3. Create task for clinical lead (due in 5 days)
  const dueDate = new Date();
  dueDate.setDate(dueDate.getDate() + 5);
  const task = await create_task({
    assignee: "clinical_lead",
    title: `Clinical inquiry — ${extracted.child_name ?? "Patient"}`,
    due: dueDate.toISOString(),
    notes: `Parent has a clinical or developmental question. Please review and provide guidance.`,
  });

  return {
    missing_info: [],
    recommended_next_action: "Clinical lead review of developmental/clinical question.",
    draft_reply: msg.data.draft_id,
    task_ids: [task.data.task_id],
    escalation: null,
    decision_rationale: "Clinical or developmental question. Per policy, automated systems do not provide advice. Routed to clinical lead.",
  };
}

// ── BRANCH: MISSING PAPERWORK ────────────────────────────────────────────────

export async function runMissingPaperworkBranch(
  item: InboxItem,
  extracted: ExtractedIntake,
): Promise<BranchResult> {
  // 1. Policy check
  await lookup_policy({ topic: "service_lines" });

  // 2. Create task for intake
  const task = await create_task({
    assignee: "intake",
    title: `Missing paperwork follow-up — ${extracted.child_name ?? "Patient"}`,
    due: new Date().toISOString(),
    notes: `Received a referral (ID: ${item.id}) but critical information is missing. Please follow up with the referring provider.`,
  });

  // 3. Draft message to referring provider
  const msg = await draft_message({
    recipient: "parent",
    channel: "email",
    body: "Triage draft: We received a referral but some critical information is missing. Our team will follow up to collect the remaining details.",
  });

  return {
    missing_info: [],
    recommended_next_action: "Intake team follow-up for missing referral details.",
    draft_reply: msg.data.draft_id,
    task_ids: [task.data.task_id],
    escalation: null,
    decision_rationale: "Referral received with missing critical fields. Routed to intake for manual follow-up.",
  };
}

// ── BRANCH: DEAD END ─────────────────────────────────────────────────────────

export async function runDeadEndBranch(
  item: InboxItem,
  classification: Classification,
): Promise<BranchResult> {
  return {
    missing_info: [],
    recommended_next_action: "No automated action. Front desk may review if needed.",
    draft_reply: null,
    task_ids: [],
    escalation: null,
    decision_rationale: `Item classified as ${classification}. No automated workflow triggered.`,
  };
}

// ── MAIN AGENT LOOP ──────────────────────────────────────────────────────────

export async function runAgent(inbox: InboxItem[]): Promise<ItemOutput[]> {
  const outputs: ItemOutput[] = [];

  for (const item of inbox) {
    const output = await withItemContext(item.id, async () => {
      // 1. Quick signal scan (per requested flow)
      const isSafeguarding = hasSafeguardingSignal(item.body);

      // 2. Structured extraction & classification
      const result = await classifyWithLLM(item);
      const classification = (isSafeguarding && result.classification !== "safeguarding")
        ? "safeguarding"
        : result.classification;
      const urgency = classification === "safeguarding" ? "P0" : result.urgency;
      const { extracted_intake: extracted } = result;

      // 3. Patient context lookup
      const searchRes = await search_patient({
        name: extracted.child_name ?? undefined,
        dob: extracted.dob_or_age ?? undefined,
      });
      const patientResult = searchRes.data.length > 0 ? { patient_id: searchRes.data[0].patient_id } : null;

      // 4. Branching based on tier
      const tier = TIER_MAP[classification];
      let branch: BranchResult;

      switch (tier) {
        case "risk":
          branch = await runSafeguardingBranch(item, extracted, classification, {});
          break;
        case "intake":
          if (classification === "missing_paperwork") {
            branch = await runMissingPaperworkBranch(item, extracted);
          } else {
            const isSpanish = SPANISH_RE.test(item.body);
            branch = await runIntakeBranch(item, extracted, patientResult, classification, { is_spanish: isSpanish });
          }
          break;
        case "patient_ops":
          branch = await runPatientOpsBranch(item, extracted, classification, {});
          break;
        case "clinical":
          branch = await runClinicalBranch(item, extracted, classification, {});
          break;
        default:
          branch = await runDeadEndBranch(item, classification);
          break;
      }

      // 5. Final output assembly (buildItemOutput)
      return {
        item_id: item.id,
        classification,
        urgency,
        requires_human_review: requiresHumanReview(
          classification,
          urgency,
          branch.missing_info,
          branch.escalation,
        ),
        extracted_intake: extracted,
        missing_info: branch.missing_info,
        tools_called: getToolCallsForItem(item.id), // Verbatim tool capture
        recommended_next_action: branch.recommended_next_action,
        draft_reply: branch.draft_reply,
        task_ids: branch.task_ids,
        escalation: branch.escalation,
        decision_rationale: branch.decision_rationale,
      } satisfies ItemOutput;
    });

    outputs.push(output);
  }

  return outputs;
}
