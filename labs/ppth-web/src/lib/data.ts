export type AgentId =
  | "dr_house"
  | "dr_foreman"
  | "dr_wilson"
  | "dr_chase"
  | "dr_cameron"
  | "dr_cuddy"
  | "case_reviewer";

export type Agent = {
  id: AgentId;
  name: string;
  short: string;
  initials: string;
  role: string;
  domain: string;
  quote: string;
  owns: string[];
  tools: string[];
  color: string;
};

export const HOUSE: Agent = {
  id: "dr_house",
  name: "Dr Gregory House",
  short: "House",
  initials: "GH",
  role: "Head of diagnostics · orchestrator",
  domain: "Diagnostic medicine",
  quote: "Everybody lies. So I read the chart.",
  owns: ["Chart retrieval", "Case framing", "Routing"],
  tools: ["get_patient_record", "differential_diagnosis"],
  color: "#0b1b2b",
};

export const COLLABORATORS: Agent[] = [
  {
    id: "dr_foreman",
    name: "Dr Eric Foreman",
    short: "Foreman",
    initials: "EF",
    role: "Neurology",
    domain: "Above the neck",
    quote: "Before we chase zebras, we rule out the horse.",
    owns: ["Seizure", "Aphasia", "Paralysis", "Stroke", "Meningitis"],
    tools: ["differential_diagnosis"],
    color: "#2a6df4",
  },
  {
    id: "dr_wilson",
    name: "Dr James Wilson",
    short: "Wilson",
    initials: "JW",
    role: "Oncology",
    domain: "Cancer & palliative care",
    quote: "Tissue is the diagnosis. Imaging is a hypothesis.",
    owns: ["Masses", "B symptoms", "Lymphoma", "LDH", "Palliative"],
    tools: ["lab_reference"],
    color: "#8b5cf6",
  },
  {
    id: "dr_chase",
    name: "Dr Robert Chase",
    short: "Chase",
    initials: "RC",
    role: "Cardiology · ICU",
    domain: "What kills fastest",
    quote: "Airway, breathing, circulation, then the puzzle.",
    owns: ["Arrhythmia", "Shock", "Troponin", "INR", "ICU & theatre"],
    tools: ["lab_reference"],
    color: "#ef4f6a",
  },
  {
    id: "dr_cameron",
    name: "Dr Allison Cameron",
    short: "Cameron",
    initials: "AC",
    role: "Immunology · infectious disease",
    domain: "Infection, immunity & drugs",
    quote: "I would rather be slow than wrong.",
    owns: ["Fever & rash", "Sepsis", "Transplant", "Drug pairs", "Formulary"],
    tools: ["check_drug_interaction", "list_formulary"],
    color: "#10b981",
  },
  {
    id: "dr_cuddy",
    name: "Dr Lisa Cuddy",
    short: "Cuddy",
    initials: "LC",
    role: "Dean of Medicine · administration",
    domain: "Anything that is not medicine",
    quote: "Approved, with conditions. The conditions are not negotiable.",
    owns: ["Consent", "Lawsuits", "Budget", "Beds", "Ethics board"],
    tools: [],
    color: "#f59e0b",
  },
  {
    id: "case_reviewer",
    name: "Case reviewer",
    short: "Reviewer",
    initials: "CR",
    role: "In-graph quality observer",
    domain: "Grades the answer, never diagnoses",
    quote: "Same answer, same score. Every time.",
    owns: ["Audit", "Critique", "Scoring", "Blocking issues"],
    tools: ["score_case_review"],
    color: "#0891b2",
  },
];

export const ALL_AGENTS: Agent[] = [HOUSE, ...COLLABORATORS];

export const byId = (id: AgentId) => ALL_AGENTS.find((a) => a.id === id)!;

export type Patient = {
  id: string;
  name: string;
  age: number;
  sex: "F" | "M";
  complaint: string;
  vitals: { label: string; value: string; flag?: boolean }[];
  history: string;
  route: AgentId;
  day: number;
};

export const PATIENTS: Patient[] = [
  {
    id: "PPTH-001",
    name: "Rebecca Adler",
    age: 29,
    sex: "F",
    complaint: "Sudden aphasia during a kindergarten class, then seizure",
    vitals: [
      { label: "BP", value: "104/68" },
      { label: "HR", value: "96" },
      { label: "Temp", value: "38.4 °C", flag: true },
      { label: "SpO₂", value: "98%" },
    ],
    history: "No significant history. Recent unpasteurised farm cheese.",
    route: "dr_foreman",
    day: 2,
  },
  {
    id: "PPTH-002",
    name: "John Henry Giles",
    age: 41,
    sex: "M",
    complaint: "Progressive weakness, now ventilator dependent",
    vitals: [
      { label: "BP", value: "132/84" },
      { label: "HR", value: "88" },
      { label: "RR", value: "12 vent", flag: true },
      { label: "SpO₂", value: "95%" },
    ],
    history: "Marathon runner. Diagnosed elsewhere with ALS two years ago.",
    route: "dr_foreman",
    day: 9,
  },
  {
    id: "PPTH-003",
    name: "Carly Forlano",
    age: 34,
    sex: "F",
    complaint: "Fever, migrating joint pain and a spreading rash",
    vitals: [
      { label: "BP", value: "118/72" },
      { label: "HR", value: "104", flag: true },
      { label: "Temp", value: "39.1 °C", flag: true },
      { label: "SpO₂", value: "97%" },
    ],
    history: "Renal transplant 14 months ago, on maintenance immunosuppression.",
    route: "dr_cameron",
    day: 1,
  },
  {
    id: "PPTH-004",
    name: "Ezra Powell",
    age: 78,
    sex: "M",
    complaint: "Confusion, chest pain and intermittent syncope",
    vitals: [
      { label: "BP", value: "96/58", flag: true },
      { label: "HR", value: "148 irr", flag: true },
      { label: "RR", value: "22" },
      { label: "SpO₂", value: "92%", flag: true },
    ],
    history: "Hypertension, paroxysmal atrial fibrillation, prior TIA.",
    route: "dr_chase",
    day: 4,
  },
];

export type Scenario = {
  prompt: string;
  target: AgentId;
  tool: string;
  why: string;
};

export const SCENARIOS: Scenario[] = [
  {
    prompt: "Patient PPTH-001 had sudden aphasia and then a seizure, with a fever. What is causing it?",
    target: "dr_foreman",
    tool: "differential_diagnosis",
    why: "Aphasia and seizure are nervous-system findings. They go to neurology, even though House could reason about them himself.",
  },
  {
    prompt: "PPTH-003 has drenching night sweats, lost weight, and an LDH of 480. Lymphoma?",
    target: "dr_wilson",
    tool: "lab_reference",
    why: "B symptoms plus a tumour-marker value. Wilson checks the LDH against the hospital reference range, never from memory.",
  },
  {
    prompt: "PPTH-004 is confused, HR 148 irregular, BP 96/58, troponin 0.9. What do we do first?",
    target: "dr_chase",
    tool: "lab_reference",
    why: "A haemodynamic emergency. Chase opens with the immediate risk and the stabilising measure.",
  },
  {
    prompt: "PPTH-004 is on warfarin. Cardiology wants to add amiodarone. Is that safe?",
    target: "dr_cameron",
    tool: "check_drug_interaction",
    why: "Any question that names two drugs goes to Cameron. House does not answer drug questions.",
  },
  {
    prompt: "The family of PPTH-002 is refusing an LP and threatening to sue.",
    target: "dr_cuddy",
    tool: "no tool needed",
    why: "Consent, refusal and legal exposure are not medicine. Cuddy authorises or refuses; she never diagnoses.",
  },
  {
    prompt: "Review and grade the answer we just gave on PPTH-001.",
    target: "case_reviewer",
    tool: "score_case_review",
    why: "An answer already exists and needs auditing. The reviewer scores it against the rubric and returns blocking issues.",
  },
];

export const TOOLS = [
  { name: "get_patient_record", owner: "dr_house", blurb: "Reads the chart first. Always. No invented vitals." },
  { name: "differential_diagnosis", owner: "dr_house · dr_foreman", blurb: "Ranked candidates, red flags first, one next test each." },
  { name: "lab_reference", owner: "dr_wilson · dr_chase", blurb: "Reference ranges from the hospital laboratory system." },
  { name: "check_drug_interaction", owner: "dr_cameron", blurb: "Checks a named drug pair against the formulary." },
  { name: "list_formulary", owner: "dr_cameron", blurb: "What the hospital actually stocks, when a drug is missing." },
  { name: "score_case_review", owner: "case_reviewer", blurb: "Deterministic score out of 100 for any proposed answer." },
];

export const SKILLS = [
  {
    name: "diagnostics-protocol",
    owner: "dr_house",
    blurb: "Chart before theory. Differential before diagnosis. Red flags precede the puzzle.",
  },
  {
    name: "case-review-rubric",
    owner: "case_reviewer",
    blurb: "Five booleans, fixed weights. The score never depends on mood.",
  },
];
