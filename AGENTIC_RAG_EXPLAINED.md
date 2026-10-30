# Agentic RAG, Explained Simply

This guide explains **Agentic RAG** in plain English. We start from the basics and build up one step at a time. No heavy math, no jargon without an explanation.

---

## Table of Contents

1. [The Big Picture in One Sentence](#1-the-big-picture-in-one-sentence)
2. [First, What Is an LLM?](#2-first-what-is-an-llm)
3. [The Problem: LLMs Don't Know Everything](#3-the-problem-llms-dont-know-everything)
4. [What Is RAG?](#4-what-is-rag)
5. [How Normal RAG Works, Step by Step](#5-how-normal-rag-works-step-by-step)
6. [Where Normal RAG Falls Short](#6-where-normal-rag-falls-short)
7. [What Is an "Agent"?](#7-what-is-an-agent)
8. [So, What Is Agentic RAG?](#8-so-what-is-agentic-rag)
9. [How Agentic RAG Works, Step by Step](#9-how-agentic-rag-works-step-by-step)
10. [A Full Example](#10-a-full-example)
11. [The Main Building Blocks](#11-the-main-building-blocks)
12. [Common Types of Agentic RAG](#12-common-types-of-agentic-rag)
13. [Normal RAG vs Agentic RAG](#13-normal-rag-vs-agentic-rag)
14. [Benefits](#14-benefits)
15. [Downsides and Things to Watch Out For](#15-downsides-and-things-to-watch-out-for)
16. [When Should You Use It?](#16-when-should-you-use-it)
17. [Simple Code Sketch](#17-simple-code-sketch)
18. [Glossary](#18-glossary)
19. [Summary](#19-summary)

---

## 1. The Big Picture in One Sentence

> **Agentic RAG is an AI that doesn't just look things up once — it thinks about what it needs, decides where to look, checks what it found, and keeps going until it has a good answer.**

Keep that sentence in mind. Everything below just explains it in more detail.

---

## 2. First, What Is an LLM?

An **LLM (Large Language Model)** is the "brain" behind tools like ChatGPT or Claude. It has read a huge amount of text and learned how language works. You give it a question, and it writes an answer.

**Analogy:** Think of an LLM as a **very well-read student**. They have read millions of books, so they can talk about almost anything. But they are answering *from memory*.

---

## 3. The Problem: LLMs Don't Know Everything

Our well-read student has some problems:

| Problem | What it means | Example |
|---|---|---|
| **Out of date** | They stopped reading at some point in the past. | They don't know yesterday's news. |
| **No private knowledge** | They never read *your* documents. | They don't know your company's leave policy. |
| **Making things up** | When unsure, they sometimes guess confidently. This is called a **hallucination**. | They invent a book title that doesn't exist. |

So we need a way to give the student the **right information at the right time**.

---

## 4. What Is RAG?

**RAG** stands for **Retrieval-Augmented Generation**. Let's break the name down:

- **Retrieval** = finding / fetching information
- **Augmented** = added to, improved
- **Generation** = writing the answer

So RAG means: **"Find useful information first, then use it to write a better answer."**

**Analogy: The open-book exam** 📖

- A normal LLM is a student taking a **closed-book exam** — answering only from memory.
- A RAG system is a student taking an **open-book exam** — they can look up the textbook before writing the answer.

The answer is now based on real material, not just memory.

---

## 5. How Normal RAG Works, Step by Step

There are two phases: **getting ready** (done once) and **answering** (done for every question).

### Phase A: Getting ready (preparing the library)

1. **Collect documents** — PDFs, web pages, notes, manuals, etc.
2. **Cut them into small pieces** called **chunks** (for example, a paragraph each). Small pieces are easier to search.
3. **Turn each chunk into numbers** called an **embedding**. An embedding captures the *meaning* of the text, so chunks about similar topics get similar numbers.
4. **Store them** in a special database called a **vector database**.

**Analogy:** This is like a librarian putting every page of every book onto index cards and arranging the cards by topic, so related cards sit close together.

### Phase B: Answering a question

1. The user asks a question.
2. The question is also turned into an embedding (numbers).
3. The system finds the chunks whose numbers are **closest** to the question — i.e., the most related pieces of text.
4. Those chunks are given to the LLM along with the question.
5. The LLM writes an answer using those chunks.

```
Question ──► Search the library ──► Top matching chunks ──► LLM writes answer
```

This is a **straight line**. One search, one answer. No going back.

---

## 6. Where Normal RAG Falls Short

Normal RAG is like a helper who follows instructions very literally:
**"Search once, grab the top results, answer. Done."**

That causes problems:

1. **Bad search, bad answer.** If the first search finds the wrong pages, the answer is wrong — and the system doesn't notice.
2. **Can't handle multi-part questions.** "Compare our 2023 and 2024 sales, and explain why they changed" needs several searches. Normal RAG does just one.
3. **Only one source.** It usually searches one library. It can't decide, "This question needs a web search" or "This needs a calculator."
4. **No checking.** It never asks itself, "Does this information actually answer the question?"
5. **Searches even when it doesn't need to.** "What is 2 + 2?" doesn't need a document search, but normal RAG does one anyway.

**Analogy:** Imagine asking an intern to research something. They go to the library, grab the first three books they see, and hand you a summary — even if those books are about the wrong topic. That's normal RAG.

---

## 7. What Is an "Agent"?

An **agent** is an AI that can **think, decide, and take actions** to reach a goal, instead of just replying once.

An agent usually has:

- **A brain** — the LLM that does the thinking.
- **Tools** — things it can use, like a search engine, a database, a calculator, or a website.
- **A loop** — it can repeat: *think → act → look at the result → think again*.
- **Memory** — it remembers what it has already done and found.

**Analogy: Hiring a detective instead of an intern** 🕵️

An intern does exactly one task and stops.
A detective:
- Thinks about what clues they need.
- Decides where to look (witnesses, records, the crime scene).
- Checks if the clues make sense.
- Follows new leads when something is missing.
- Only closes the case when the story fits together.

That is an agent.

---

## 8. So, What Is Agentic RAG?

Put the two ideas together:

> **Agentic RAG = RAG + an agent that is in charge of the searching.**

Instead of a fixed "search once, then answer" pipeline, an agent **controls** the whole process. It decides:

- **Do I even need to search?**
- **Where should I search?** (company documents, the web, a database, etc.)
- **What exact search should I run?** (it can rewrite the question to get better results)
- **Is what I found good enough?**
- **Do I need to search again, or search somewhere else?**
- **Am I ready to answer?**

**Analogy: A smart research assistant** 🧑‍💼

Normal RAG is a **vending machine**: press a button, get whatever is in that slot.

Agentic RAG is a **personal research assistant**: you tell them what you need, and they figure out how to get it — asking the right people, checking several sources, double-checking facts, and coming back with a well-researched answer.

---

## 9. How Agentic RAG Works, Step by Step

Here is the typical loop. Don't worry, it's simpler than it looks.

```
                ┌──────────────────────────┐
                │   User asks a question   │
                └────────────┬─────────────┘
                             ▼
                ┌──────────────────────────┐
                │ 1. THINK: What do I need? │◄──────────┐
                └────────────┬─────────────┘           │
                             ▼                         │
                ┌──────────────────────────┐           │
                │ 2. PLAN: Which tool/source│           │
                │    should I use?          │           │
                └────────────┬─────────────┘           │
                             ▼                         │
                ┌──────────────────────────┐           │
                │ 3. ACT: Search / use tool │           │
                └────────────┬─────────────┘           │
                             ▼                         │
                ┌──────────────────────────┐    No,    │
                │ 4. CHECK: Is this enough  ├───────────┘
                │    and is it correct?     │  try again
                └────────────┬─────────────┘
                             │ Yes
                             ▼
                ┌──────────────────────────┐
                │ 5. ANSWER: Write the reply│
                │    using what was found   │
                └──────────────────────────┘
```

Let's walk through each step:

### Step 1 — Think
The agent reads the question and works out what information it needs.
*"The user wants X. To answer that, I need to know A and B."*

### Step 2 — Plan
It picks the right tool or source for each piece.
*"A is in our internal documents. B is recent news, so I'll search the web."*

### Step 3 — Act
It runs the search (or calls a tool) and gets results back.

### Step 4 — Check
This is the key step that normal RAG doesn't have. The agent looks at the results and asks:
- *Are these results actually about my question?*
- *Is anything missing?*
- *Do the sources disagree?*

If something is wrong or missing, it **goes back to Step 1** — maybe with a reworded search or a different source.

### Step 5 — Answer
Once it's happy, it writes the final answer, ideally saying where the information came from.

---

## 10. A Full Example

**Question:** *"Is our company's new remote work policy more generous than the national average?"*

### With normal RAG
1. Searches company documents for "remote work policy national average".
2. Finds the company policy (good) but nothing about the national average (that isn't in company docs).
3. LLM writes an answer — and may **guess** the national average. ❌

### With Agentic RAG
1. **Think:** "I need two things: (a) our new policy, (b) the national average."
2. **Plan:** "(a) is in the company documents. (b) needs a web search."
3. **Act:** Searches company documents → finds an *old* 2022 policy.
4. **Check:** "Wait, the question says *new* policy. This is from 2022. Let me search again."
5. **Act:** Searches again for "remote work policy 2024 update" → finds the new policy: 3 remote days per week.
6. **Act:** Web search for national remote work statistics → finds the average is about 2 days per week.
7. **Check:** "I have both pieces, from reliable sources. Good."
8. **Answer:** "Yes. Our new 2024 policy allows 3 remote days per week, compared with a national average of about 2. *(Sources: HR Policy 2024, page 4; National Labor Survey 2024.)*" ✅

Notice how the agent **caught its own mistake** (the old policy) and **used two different sources**. That's the whole point.

---

## 11. The Main Building Blocks

Think of Agentic RAG as a small team. Each member has a job.

| Building block | What it does | Analogy |
|---|---|---|
| **LLM (the brain)** | Thinks, plans, decides, and writes the final answer. | The detective's mind |
| **Retriever** | Searches documents and returns relevant chunks. | The librarian |
| **Vector database** | Stores document chunks by meaning so they can be searched fast. | The organized library shelves |
| **Tools** | Extra abilities: web search, calculator, SQL database, APIs. | The detective's gadgets |
| **Router** | Decides which source or tool should handle a question. | The receptionist who sends you to the right department |
| **Query rewriter** | Rewrites a vague question into a clearer search. | A friend who says "Oh, you mean *this*?" |
| **Grader / Checker** | Judges whether the results are relevant and the answer is supported. | The editor who reviews a draft |
| **Memory** | Remembers what has been searched and found so far. | The detective's notebook |

You don't always need every piece. Simple systems might only use a few.

---

## 12. Common Types of Agentic RAG

There are a few popular "styles". You'll hear these names a lot.

### 12.1 Routing Agent (the receptionist)
The agent only decides **where** to send the question.
- Question about HR? → Search HR documents.
- Question about code? → Search the code docs.
- Simple chit-chat? → No search needed.

*Simplest form of Agentic RAG.*

### 12.2 Query Planning Agent (the project manager)
Breaks a big question into smaller ones, answers each, then combines them.
- "Compare Product A and Product B" → (1) Find info on A. (2) Find info on B. (3) Compare.

### 12.3 ReAct Agent (Reason + Act)
The agent alternates between **thinking out loud** and **taking an action**, over and over:
```
Thought: I need the 2024 sales numbers.
Action:  search("2024 sales report")
Result:  Sales were $5M.
Thought: Now I need 2023 numbers to compare.
Action:  search("2023 sales report")
Result:  Sales were $4M.
Thought: I have both. Sales grew by 25%.
Answer:  Sales grew 25%, from $4M to $5M.
```

### 12.4 Self-Correcting RAG (the careful student)
After searching, the agent **grades** its own results. If they're poor, it rewrites the question and tries again, or switches to another source (like a web search). Names you may see: **Self-RAG** and **Corrective RAG (CRAG)**.

### 12.5 Multi-Agent RAG (the team)
Several specialist agents work together, with a "manager" agent coordinating them.
- One agent searches legal documents.
- One agent searches financial data.
- One agent checks facts.
- The manager combines everything into one answer.

**Analogy:** A newsroom — reporters gather facts, a fact-checker verifies them, and the editor puts the story together.

---

## 13. Normal RAG vs Agentic RAG

| | Normal RAG | Agentic RAG |
|---|---|---|
| **Flow** | Straight line: search → answer | Loop: think → act → check → repeat |
| **Number of searches** | Usually one | As many as needed |
| **Sources** | Usually one | Many (docs, web, databases, tools) |
| **Checks its own work?** | No | Yes |
| **Handles complex questions?** | Poorly | Well |
| **Speed** | Fast | Slower |
| **Cost** | Low | Higher (more LLM calls) |
| **Predictability** | Very predictable | Less predictable |
| **Analogy** | Vending machine | Personal research assistant |

---

## 14. Benefits

1. **More accurate answers** — it checks results and retries when they're poor.
2. **Handles hard, multi-step questions** — it can break them into parts.
3. **Uses the right source** — documents, the web, databases, or tools as needed.
4. **Fewer made-up answers** — it grounds the answer in real, checked information.
5. **Smarter use of search** — it can skip searching when it's not needed.
6. **Explains itself** — it can show which sources it used.

---

## 15. Downsides and Things to Watch Out For

Agentic RAG is powerful, but it isn't free.

1. **Slower** — more thinking and more searching means more waiting.
2. **More expensive** — each "think" step is another LLM call, which costs money.
3. **Can get stuck in loops** — it might keep searching forever. *Fix:* set a maximum number of steps.
4. **Harder to debug** — because it makes its own decisions, it's harder to see why it gave a certain answer. *Fix:* log every step.
5. **Can still make mistakes** — a bad decision early on can send it in the wrong direction.
6. **More complex to build** — more moving parts means more things that can break.

**Analogy:** Hiring a detective is great for a tricky case, but you wouldn't hire one just to find your car keys in your own pocket.

---

## 16. When Should You Use It?

**Use normal RAG when:**
- Questions are simple and direct ("What is our refund policy?").
- All information lives in one place.
- Speed and low cost matter most.

**Use Agentic RAG when:**
- Questions are complex or have several parts.
- Information is spread across many sources.
- Accuracy matters a lot (legal, medical, finance, research).
- The system needs to use tools (calculators, APIs, live data).

**Tip:** Start simple. Begin with normal RAG, and add agent features (like a router or a checker) only where you see it failing.

---

## 17. Simple Code Sketch

This is **pseudocode** — simplified code to show the idea, not a finished program.

### Normal RAG
```python
def normal_rag(question):
    chunks = vector_db.search(question, top_k=3)   # search once
    answer = llm.generate(question, context=chunks) # answer
    return answer
```

### Agentic RAG
```python
def agentic_rag(question, max_steps=5):
    notes = []                                      # memory

    for step in range(max_steps):                   # stop runaway loops
        # 1. THINK + 2. PLAN: the LLM decides what to do next
        decision = llm.decide(question, notes,
                              tools=["search_docs", "search_web", "calculator", "answer"])

        if decision.tool == "answer":               # 5. ANSWER when ready
            return llm.generate(question, context=notes)

        # 3. ACT: run the chosen tool
        result = run_tool(decision.tool, decision.input)

        # 4. CHECK: is this result useful?
        if llm.is_relevant(question, result):
            notes.append(result)
        # if not relevant, the loop continues and the LLM tries something else

    # Out of steps: answer with what we have, and be honest about it
    return llm.generate(question, context=notes, note="May be incomplete.")
```

The big difference: the agentic version has a **loop**, a **choice of tools**, and a **check**.

Popular libraries that help you build this include **LangGraph**, **LangChain**, **LlamaIndex**, **CrewAI**, and the **Claude Agent SDK**.

---

## 18. Glossary

| Term | Simple meaning |
|---|---|
| **LLM** | The AI "brain" that reads and writes text. |
| **RAG** | Look up information first, then answer using it. |
| **Agent** | An AI that can think, choose actions, use tools, and repeat until done. |
| **Agentic RAG** | RAG where an agent controls the searching and checking. |
| **Chunk** | A small piece of a document (like a paragraph). |
| **Embedding** | A list of numbers that represents the meaning of text. |
| **Vector database** | A database that stores embeddings and finds similar ones quickly. |
| **Retriever** | The part that searches and returns relevant chunks. |
| **Tool** | Something the agent can use: web search, calculator, database, API. |
| **Router** | Decides which source or tool to use. |
| **Query rewriting** | Rephrasing a question to get better search results. |
| **Hallucination** | When the AI confidently makes something up. |
| **Grounding** | Basing the answer on real, retrieved information. |
| **ReAct** | A pattern where the agent alternates between Reasoning and Acting. |
| **Context** | The information given to the LLM along with the question. |

---

## 19. Summary

- An **LLM** is a smart student answering from memory — but its memory can be old, incomplete, or wrong.
- **RAG** turns it into an **open-book exam**: look things up first, then answer.
- **Normal RAG** searches **once** and answers — like a vending machine.
- An **agent** can **think, use tools, check its work, and try again** — like a detective.
- **Agentic RAG** puts an agent in charge of the research: it **decides where to look, checks what it finds, and keeps going until the answer is good**.
- It's **more accurate and flexible**, but **slower, costlier, and more complex**.
- **Start simple**, and add agentic features where they actually help.

> **Remember:** Normal RAG *looks up* an answer. Agentic RAG *researches* an answer.
