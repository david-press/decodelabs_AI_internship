# Rule-Based AI Chatbot + Calculator
### DecodeLabs Industrial Training | Batch 2026 — Project 1


---

## What This Is

A rule-based AI chatbot built entirely from scratch using pure Python logic — no machine learning, no APIs, no external libraries. Just control flow, dictionaries, and intentional design.

This project is the foundation phase of AI engineering: mastering **deterministic systems** before touching probabilistic ones. Every major AI product in production (think banking chatbots, fraud detection, medical triage systems) has a rule layer exactly like this sitting in front of the "smart" part.

---

## Core Concepts Behind This Build

| Concept | What It Means |
|--------|---------------|
| **Deterministic AI** | Same input → same output. Always. No surprises. |
| **White Box System** | Every decision is traceable: Input → Logic → Output |
| **Hash Map (Dictionary)** | O(1) lookup — instant response regardless of how many rules exist |
| **If-Elif Anti-Pattern** | The wrong approach: O(n) — slows down as rules grow |
| **Input Sanitization** | `.lower().strip()` — "Hello", "HELLO", " hello " all treated equally |
| **IPO Model** | Input → Process → Output — the skeleton of every program ever written |
| **Separation of Concerns** | Exit logic, math logic, and chat logic each live in their own layer |

---

## Project Versions

### V1 — Core Chatbot

**What it does:** Responds to predefined user inputs using a knowledge base dictionary. Runs in a continuous loop until the user exits.

**The 5 required components:**
- `while True` — infinite input loop (the heartbeat)
- `.lower().strip()` — sanitization layer
- Dictionary with 8 intents — the knowledge base
- `.get()` with fallback — graceful handling of unknown inputs
- `break` on exit commands — clean session termination

```python
# The entire response engine in one line
reply = knowledge_base.get(user_input, "I don't understand that yet.")
```

**Run it:**
```bash
python chatbot_v1.py
```

---

### V2 — Chatbot + Calculator

Added a dynamic math processing layer on top of V1. The key architectural insight: the knowledge base handles *static* inputs (same input, same output), but math inputs are *dynamic* — `"2 + 3"` and `"99 * 7"` are both math, but the response changes every time. They need a function, not a dictionary entry.

**New components:**
- `re` module for pattern detection (is this input math?)
- Input whitelisting — security layer that blocks code injection
- `ZeroDivisionError` handling
- `round()` for float precision

**The 3-layer processing pipeline:**

```
Input → Sanitize → Exit check? → Math? → Dictionary → Fallback
```

**Security note:** Raw `eval()` on user input is dangerous. A user could type `__import__('os').system('...')` and execute system commands. This version whitelists only digits, operators, and parentheses before evaluating anything.

```bash
python chatbot_v2.py
```

---

## How to Run

**Requirements:**
- Python 3.x
- No pip installs needed — only built-in Python modules used (`re`)

**Steps:**
```bash
# 1. Clone or download this repo
git clone https://github.com/yourusername/decodelabs-project1

# 2. Navigate into the folder
cd decodelabs_AI_project1

# 3. Run V1 (chatbot only)
python rule_based_chatbot_v1.py

# 4. Or run V2 (chatbot + calculator)
python rule_based_chatbot_v2.py
```

**Sample session:**
```
==================================================
   DataBot v2.0 — Chatbot + Calculator
   Type 'exit' to quit.
==================================================

You: hello
Bot: Hey! Welcome to DataBot. How can I help?

You: 25 * 4
Bot: = 100

You: (3 + 5) * 2
Bot: = 16

You: 10 / 0
Bot: Error: You can't divide by zero.

You: bye
Bot: Goodbye! Session ended.
```

---

## Edge Cases Handled

- Mixed case input: `"Hello"` `"HELLO"` `"hello"` → all matched correctly
- Whitespace: `"  hello  "` → stripped and matched
- Unknown input → graceful fallback, no crash
- Division by zero → caught and handled
- Float imprecision: `0.1 + 0.2` → rounded to `0.3`
- Code injection attempt: `__import__('os')` → blocked by whitelist
- Empty input (just pressing Enter) → fallback response

---

## Tech Stack

```
Language   →  Python 3
Modules    →  re (built-in, regex pattern matching)
Concepts   →  Control flow, hash maps, input sanitization,
              regex, security whitelisting, algorithmic complexity
```

---

## What This Connects To

This isn't just an exercise. The pattern built here — a deterministic rule layer that sits in front of a probabilistic system — is the exact architecture used by:

- **NVIDIA NeMo Guardrails** — rule-based safety layer for LLMs
- **Bank fraud detection** — hard rules before ML models
- **Customer support bots** — fast rule matching before AI escalation

The foundation is always the logic engine. The intelligence comes after.

---

*Built as part of DecodeLabs AI Industrial Training Program, Batch 2026.*
