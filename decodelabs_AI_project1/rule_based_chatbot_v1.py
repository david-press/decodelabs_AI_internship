# ================================================================
# VERSION 1 (Chatbot Only)
# ================================================================

# ============================================================
# DECODELABS PROJECT 1: Rule-Based AI Chatbot
# ============================================================

def get_response(user_input , knowledge_base):
    """
    Looks up user_input in the knowledge_base dictionary.
    Returns the matching response, or a fallback message.
    
    .get(key, default) returns value if key exists,
    otherwise returns the default string
    """
    return knowledge_base.get(user_input , "I don't understand")


def sanitize(raw_input):
    """
    Normalizes user input so our matching works reliably.
    
    .lower()  converts every input to lower case
    .strip()  removes  leading/trailing spaces

    """
    return raw_input.lower().strip()

def run_chatbot():
    """
    The main chatbot function.
    Contains the knowledge base and the infinite input loop.
    """
     
    knowledge_base = {
        "hello" : "Hey! welcome to Databot. How can I help you today?",
        "hi" : "Hi there! What can I help you with?",
        "how are you" : "I am a bot, so I'm always running 100% , How about you?",
        "what is ai" : "AI is the science of making machines simulate human intellect using data and logic",
        "what can you do" : "I can answer simple basic questions, type help to see my commands",
        "help" : "Available commands : hello , hi , how are you , what is ai , what can you do , bye , exit",
        "thanks" : "You are welcome",
        "who made you" : "I was built by an AI engineer intern at DecodeLabs!"
       
        
    }

    # ---------------------------------------------------------
    # STARTUP MESSAGE
    # ---------------------------------------------------------
    print("=" * 50)
    print("     DataBot v1.0 - Rule-Based AI Chatbot     ")
    print("=" * 50)


    # ---------------------------------------------------------
    # THE INFINITE LOOP — The Heartbeat
    # while True runs forever until 'break' is hit
    # ---------------------------------------------------------

    while True:
        raw_input = input("\nYou: ")
        
        clean_input = sanitize(raw_input)

        if clean_input in ["exit" , "quit" , "bye" , "goodbye"]:
            print("\nBot: Goodbye. Session ended.")
            break


        response = get_response(clean_input , knowledge_base)
        print (f"Bot: {response}")



# ---------------------------------------------------------
# ENTRY POINT
# Only runs if this file is executed directly (not imported)
# ---------------------------------------------------------
if __name__ == "__main__":
    run_chatbot()