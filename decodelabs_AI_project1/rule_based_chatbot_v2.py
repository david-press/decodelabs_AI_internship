# ================================================================
# VERSION 2 (Chatbot + Calculator)
# ================================================================

# ============================================================
# DECODELABS PROJECT : Rule-Based AI Chatbot + Calculator
# ============================================================

import re
# 're' is Python's built-in regex (Regular Expression) module.
#Used to DETECT if the user's input looks like math.
# Regex lets us search for patterns inside strings.


def sanitize(raw_input):
    return raw_input.lower().strip()


def is_math(user_input):
    """
    Detects if the input looks like a math expression.
    
    re.search(pattern, string) scans the string for the pattern.
    Returns a match object if found, None if not.
    
    The pattern r'[\d]' means: "does this string contain
    at least one digit (0-9)?"
    
    Examples:
      "2 + 3"  → contains digits → True
      "hello"  → no digits       → False
      "10 * 5" → contains digits → True
    """
    return bool(re.search(r'[\d]', user_input))


def calculate(expression):
    """
    Safely evaluates a math expression string.
    
    The UNSAFE way: eval(expression)
    eval() executes ANY Python code — a user could type
    "__import__('os').system('rm -rf /')" and destroy files.
    NEVER use raw eval() on user input in real projects.
    
    The SAFE way: whitelist exactly what's allowed.
    We check the expression contains ONLY:
      - digits (0-9)
      - operators (+ - * / %)
      - decimal points, spaces, parentheses
    
    If anything else is found, we reject it entirely.
    """

    # Whitelist check — only allow safe math characters
    # r'[^\d\s\+\-\*\/\%\.\(\)]' means:
    # "find any character that is NOT a digit, space, operator,
    #  decimal point, or parenthesis"
    if re.search(r'[^\d\s\+\-\*\/\%\.\(\)]', expression):
        return "I can only calculate numbers and operators: + - * / %"

    try:
        # eval() is safe here because we already whitelisted input
        result = eval(expression)

        # Round to 4 decimal places to avoid floating point ugliness
        result = round(result, 4)

        return f" {result}"

    except ZeroDivisionError:
        return "Error: You can't divide by zero."

    except Exception:
        return "I couldn't calculate that. Try something like: 10 + 5"


def get_response(user_input, knowledge_base):
    return knowledge_base.get(
        user_input,
        "I don't understand that yet. Type 'help' to see what I know."
    )


def run_chatbot():

    knowledge_base = {
        "hello": "Hey! Welcome to DataBot. How can I help?",
        "hi": "Hi there! What can I do for you?",
        "hey" : "Hi there! What can I do for you?",
        "how are you": "Always running at 100%! How about you?",
        "what is ai": "AI is the science of making machines simulate human intelligence.",
        "what can you do": "I can chat and calculate! Try: 25 * 4 or type 'help'.",
        "help": "Commands: hello, how are you, what is ai\nMath: just type it e.g. 100 / 4 ",
        "thanks": "You're welcome!",
        "who made you":"Built by an AI engineer intern at DecodeLabs!",
    }

    print("=" * 50)
    print("   DataBot v2.0 — Chatbot + Calculator")
    print("   Type 'exit' to quit.")
    print("=" * 50)

    while True:
        raw = input("\nYou: ")
        clean_input = sanitize(raw)

        # Layer 1: Exit check
        if clean_input in ["exit", "quit", "bye", "goodbye"]:
            print("\nBot: Goodbye! Session ended.")
            break

        # Layer 2: Math check — runs BEFORE dictionary lookup
        # Dynamic inputs (math) handled by a function,
        # Static inputs (chat) handled by the dictionary
        if is_math(clean_input):
            result = calculate(clean_input)
            print(f"Bot: {result}")
            continue
            # 'continue' skips rest of this loop iteration
            # and jumps back to the top (the input() line)

        # Layer 3: Dictionary lookup for chat responses
        response = get_response(clean_input, knowledge_base)
        print(f"Bot: {response}")


if __name__ == "__main__":
    run_chatbot()

