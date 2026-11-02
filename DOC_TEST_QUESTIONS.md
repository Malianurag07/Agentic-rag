# Test questions for the policy-manual documents

50 questions for `Rag testing doc.pdf`, `enterprise_company_policy_manual.docx` and `enterprise_company_policy_manual.txt` (one 30-page manual in three formats). Machine-readable version: [`data/doc_questions.jsonl`](data/doc_questions.jsonl).

Each question should be run against **all three formats**, and the answer should be the same in each.

**How the answers were checked.** Every expected fact below was verified to exist in the stored text of the PDF, the DOCX and the TXT (`tests/test_doc_questions.py`). If a question fails later, the answer key is not the cause.

**Reading the table.** *Where* is the manual subsection that holds the answer. `a / b` means either of those subsections is acceptable (the golden rules are repeated in several). `a + b` means the answer needs both subsections.

| Kind | Count |
|---|---|
| prose lookup | 10 |
| table lookup | 12 |
| multi-part | 12 |
| golden rule | 4 |
| paraphrase | 2 |
| typo | 1 |
| vague | 1 |
| follow-up | 2 |
| unanswerable | 3 |
| trap | 1 |
| chart-only | 2 |

## A. Single fact in prose

| # | Question | Expected answer | Where |
|---|---|---|---|
| 01 | How long is the probationary period for new employees? | 90 calendar days; up to 180 days for Grade L6+ leadership roles. | 5.1 |
| 02 | How many days do I have to appeal a disciplinary decision? | 7 business days from receiving the decision, to be written as a Notice of Appeal. | 14.3 |
| 03 | Who is allowed to speak to the media on behalf of the company? | Only the CEO, CFO, General Counsel and designated Corporate Communications officers; everyone else refers journalists to press-office@enterprise.internal. | 12.3 |
| 04 | How long is the pre-publication review for a technical blog post that mentions company architecture? | A mandatory ten-day review via the IP & Communications Portal. | 12.3 |
| 05 | By when does the company aim to reach net-zero for Scope 1 and Scope 2 emissions? | By 2030 (and a 50% cut in Scope 3 emissions intensity by 2035 against a 2022 baseline). | 15.1 |
| 06 | What internal carbon price must be used for capital proposals above $100,000? | A shadow internal carbon price of $85 per metric ton of CO2-equivalent. | 15.1 |
| 07 | How quickly must I forward a data subject access request to the privacy team? | Within twelve (12) hours, to privacy-dsar@enterprise.internal. | 4.4 |
| 08 | How many times a year can an employee look at their own HR personnel file? | Twice per calendar year, via the Employee Self-Service portal. | 5.4 |
| 09 | How long can a knife blade be before it is banned on company premises? | Blades exceeding three (3) inches are prohibited. | 2.4 |
| 10 | How soon must a workplace injury or near-miss be logged? | Within twelve (12) hours, in the EHS Portal. | 9.4 |

## B. Lookups inside tables

| # | Question | Expected answer | Where |
|---|---|---|---|
| 11 | What encryption standard is required for data at rest on laptops? | AES-256-GCM full-disk encryption (BitLocker / FileVault / LUKS). | 3.2 |
| 12 | What is the patch deadline for a critical vulnerability with a CVSS score of 9.0 or higher? | 48 hours. | 3.2 |
| 13 | What is the longest a just-in-time privileged access session can last? | 4 hours (automatic session revocation at the 240-minute mark). | 3.2 |
| 14 | Who has to approve a purchase of $60,000? | Tier 3: a Vice President (Grade L8+), the FP&A Director and the Procurement Lead. | 6.2 |
| 15 | What approvals does a $600,000 commitment need? | Tier 5: the CEO, the CFO and a Board Audit Committee Resolution. | 6.2 |
| 16 | How many written quotes are needed for a purchase between $10,000 and $49,999? | At least two (2) written competitive quotes. | 8.2 |
| 17 | How many days of paid annual leave do employees with five or more years of service get? | 27 days (22 business days for 0 to 4 years of tenure). | 10.2 |
| 18 | How many weeks of parental leave does a birth parent get? | 20 weeks at 100% base salary (12 weeks for the secondary, adoptive or surrogacy parent). | 10.2 |
| 19 | What is the hotel price ceiling per night in a top-tier global city like London? | $350 per night, excluding tax. | 11.2 |
| 20 | How long are personnel records kept? | 7 years after termination (record series HR-03). | 4.3 |
| 21 | How fast must a whistleblower report be acknowledged? | Within 24 hours of receipt (Phase 1: Intake & Acknowledgment). | 13.2 |
| 22 | What data-center efficiency and renewable energy targets does the company set? | PUE at or below 1.18 and at least 85% renewable matching. | 15.2 |

## C. Multi-part questions that need more than one section

| # | Question | Expected answer | Where |
|---|---|---|---|
| 23 | What are the rules for submitting expense claims, and what per-day meal allowance applies when travelling in a Tier-1 city? | Claims must be submitted within 30 calendar days with an itemised original receipt (claims over 60 days are forfeited or taxed). The Tier-1 city meal per diem is $95/day. | 6.3 + 11.2 |
| 24 | What is the limit on gifts I can accept, and how early must I disclose outside employment? | Gifts under $100 USD may be accepted without approval (over $100 must be surrendered and logged). Outside employment must be disclosed at least 15 business days before accepting it. | 11.3 + 1.3 |
| 25 | How do sick leave and bereavement leave differ in the number of days allowed? | Sick and mental health leave is 12 business days per year; bereavement is 10 business days for immediate family (3 days for extended family). | 10.2 |
| 26 | What counts as a Category A offense and what is the sanction for it? | Gross misconduct, such as failing to report a data breach within 24 hours, retaliation, fraud, harassment, theft or bribery. Sanction: suspension, then summary termination for cause plus equity clawback. | 14.2 |
| 27 | If I report a safety hazard, am I protected from retaliation, and how is that checked after an investigation? | Yes, the zero-tolerance anti-retaliation guarantee covers it; the Ombudsman runs 90-day and 180-day post-closure audits. | 1.2/2.1/9.1/13.1 + 13.2 |
| 28 | What are the remote work tiers, and what tax risk comes with working from another country? | Tiers A to D (e.g. Tier B Standard Hybrid is 3 days a week in office). Working abroad can create Permanent Establishment corporate tax nexus. | 7.2 + 7.3 |
| 29 | How fast must a lost company laptop be reported, and what discipline can follow if it is not? | Within 24 hours of discovery, to ISOC and Legal Compliance; failing to do so is a Category-A disciplinary offense. | 3.1/4.1/7.1/13.1 |
| 30 | How many days a year can I work from another country, and who must clear it? | Up to 20 business days a year (Tier D); written clearance is needed from the Global Mobility & Corporate Tax Directorate. | 7.2 + 7.3 |
| 31 | What approvals does a $300,000 vendor contract need, and what sourcing process applies to it? | Tier 4 approval by a C-Suite Functional Officer, the CFO and Legal; sourcing by formal sealed-bid RFP. | 6.2 + 8.2 |
| 32 | Which open-source licences are prohibited, and which data levels may be used with the enterprise AI sandbox? | Strong network copyleft (GPL, AGPL, SSPL) is prohibited in proprietary products; enterprise-hosted GenAI sandboxes are permitted for Level 1, 2 and 3 data. | 12.2 |
| 33 | What are the information classification levels, and how long are security logs kept? | Four tiers, from Level 4 Restricted down to Level 1 Public; security and audit logs are kept 3 years (1 year hot, 2 years cold). | 4.2 + 4.3 |
| 34 | How long does an internal investigation take, and how long do I have to appeal the outcome? | Determination is due in days 26 to 35; a Notice of Appeal must be filed within 7 business days and the panel rules within 10 business days. | 13.2 + 14.3 |

## D. The five repeated golden rules

| # | Question | Expected answer | Where |
|---|---|---|---|
| 35 | In which sections is the 24-hour breach reporting rule repeated? | Sections 3, 4, 7 and 13 (4 locations). | 18.2 |
| 36 | Who is the audit owner of the Four-Eyes dual financial authorization rule? | The Chief Financial Officer & Internal Audit Directorate. | 18.2 |
| 37 | After how many minutes of inactivity must an endpoint lock automatically? | No more than five (5) minutes. | 3.1/4.1/7.1 |
| 38 | By what date must the annual compliance recertification be done, and what happens if I am late? | Before December 15 each year; more than 14 calendar days late means automatic suspension of network credentials and badge access. | 1.2/5.1/14.1 |

## E. Different wording

| # | Question | Expected answer | Where |
|---|---|---|---|
| 39 | Can I take my work laptop abroad on holiday and keep working? | Not without prior written clearance from the Global Mobility & Corporate Tax Directorate. | 7.3 |
| 40 | Is it allowed for a manager to be in a relationship with someone on their own team? | It is classed as 'Fraternization in Reporting Line': it must be disclosed and the result is mandatory reassignment (sanction if undisclosed). | 2.3/2.4 |

## E. Typos

| # | Question | Expected answer | Where |
|---|---|---|---|
| 41 | what is the pasword length rule for service acounts | 24 characters for service accounts (16 for standard accounts). | 3.2 |

## E. Vague question

| # | Question | Expected answer | Where |
|---|---|---|---|
| 42 | leave | **Clarifying question or answer.** Either a clarifying question or the leave entitlement schedule (Section 10.2); a guess about one leave type would be wrong. | 10.2 |

## E. Follow-up questions

| # | Question | Expected answer | Where |
|---|---|---|---|
| 43 | and for sick leave? *(after: "How many days of paid annual leave do employees with five or more years of service get?")* | 12 business days per calendar year at full pay. | 10.2 |
| 44 | what about the second tier cities? *(after: "What is the hotel price ceiling per night in a top-tier global city like London?")* | Tier-2 metros: $220 per night. | 11.2 |

## F. Not answerable from the documents

| # | Question | Expected answer | Where |
|---|---|---|---|
| 45 | What is the company's policy on pet insurance? | **Should be refused.** Not covered anywhere in the manual; the system should say so. | none (should be refused) |
| 46 | What is the current share price of the company? | **Should be refused.** Not in the manual. | none (should be refused) |
| 47 | What is the name of the current CEO? | **Should be refused.** The manual names the role but never a person. | none (should be refused) |

## F. Trap question

| # | Question | Expected answer | Where |
|---|---|---|---|
| 48 | What is the dress code on Fridays? | **Should be refused.** Section 2.4's title mentions 'Dress Code Standards' but the text never defines a dress code, so the correct answer is that it is not specified. | none (should be refused) |

## F. Answer exists only inside a chart image

| # | Question | Expected answer | Where |
|---|---|---|---|
| 49 | How many auditable control checkpoints does the Cyber & Data domain have? | **Should be refused.** The figure is only inside a chart image (48 of 212). It is not in the text of any of the three files, so a text-only system cannot answer. | none (should be refused) |
| 50 | What is the review cycle in days for a Tier 5 procurement approval? | **Should be refused.** The 30-day figure is only in a chart image; it is not in the text of any of the three files. | none (should be refused) |

## What a good run looks like

- Questions A to D: the answer contains the listed facts and cites the right section.
- Multi-part (C): the answer covers **every** part; an answer that covers only one part is a fail.
- Golden rules (D): any one of the repeated locations is a correct source.
- Different wording, typos, follow-ups (E): same facts as the plain version of the question.
- F: the system must say it cannot find a reliable answer. A confident invented answer is the worst outcome, especially for the dress-code trap (a section title mentions it but the text never defines one) and the two chart-only questions (the numbers exist only inside images, so a text-only system cannot know them).
