"""
FIX FOR COLAB - Run this cell after cloning the repo to fix agents.py
This fixes the KeyError: 'answer' issue and prevents infinite loops
"""

import os
import re

# Read agents.py
with open('agents.py', 'r') as f:
    agents_code = f.read()

# FIX 1: Replace the query method completely
new_query_method = '''    def query(self, question: str):
        """Query the RAG system - FIXED VERSION."""
        initial_state = {
            "question": question,
            "documents": "",
            "images": [],
            "chat_history": [],
            "image_query_triggered": False,
            "answer": ""  # Initialize answer field
        }
        
        try:
            # Use invoke() to get final merged state (not stream())
            final_state = self.graph.invoke(initial_state)
            
            # Extract answer - handle different state structures
            if isinstance(final_state, dict):
                # Method 1: Direct answer field (from invoke)
                answer = final_state.get("answer", "")
                if answer and answer.strip() and len(answer.strip()) > 10:
                    return answer.strip()
                
                # Method 2: Nested under node name (from stream or node output)
                if "answer_generator" in final_state:
                    nested = final_state["answer_generator"]
                    if isinstance(nested, dict):
                        nested_answer = nested.get("answer", "")
                        if nested_answer and nested_answer.strip() and len(nested_answer.strip()) > 10:
                            return nested_answer.strip()
                
                # Method 3: From messages
                if "messages" in final_state:
                    messages = final_state["messages"]
                    for msg in reversed(messages):
                        if hasattr(msg, "content") and msg.content:
                            content = str(msg.content).strip()
                            if len(content) > 10:
                                return content
                        elif isinstance(msg, str) and len(msg.strip()) > 10:
                            return msg.strip()
            
            return "No answer generated. Please try rephrasing your question."
            
        except Exception as e:
            error_msg = str(e)
            print(f"Error in query: {error_msg[:200]}")
            import traceback
            traceback.print_exc()
            return f"Error: {error_msg[:200]}. Please check your configuration and try again."
'''

# FIX 2: Replace generate_answer to prevent infinite loops
new_generate_answer = '''        def generate_answer(state):
            """Generate the final answer - FIXED VERSION."""
            question = state.get("question", "")
            documents = state.get("documents", "")
            images = state.get("images", [])
            chat_history = state.get("chat_history", [])
            
            # CRITICAL: Prevent infinite loops
            # Check if we're repeating the same question
            if chat_history:
                last_entries = chat_history[-3:] if len(chat_history) >= 3 else chat_history
                question_count = sum(1 for entry in last_entries if entry[0] == "user" and entry[1] == question)
                if question_count >= 2:
                    return {
                        "answer": "I notice this question was already asked. Please try rephrasing or ask a different question.",
                        "chat_history": chat_history
                    }
            
            # Limit chat history to prevent token overflow
            if len(chat_history) > 6:
                chat_history = chat_history[-6:]  # Keep only last 6 entries
            
            # If using OpenAI and images are available, use vision model
            if self.primary_llm_type == "openai" and images:
                try:
                    image_messages = [{
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img}"}
                    } for img in images[:2]]  # Limit to 2 images
                    
                    vision_content = [
                        {"type": "text", "text": f"Context: {documents}\\nQuestion: {question}"},
                        *image_messages
                    ]
                    response = self.vision_llm.invoke([("user", vision_content)])
                    answer = response.content if hasattr(response, "content") else str(response)
                except Exception as e:
                    print(f"Vision model error: {e}, falling back to text")
                    answer_chain = self.answer_agent
                    response = answer_chain.invoke({"question": question, "documents": documents, "images": []})
                    answer = response.content if hasattr(response, "content") else str(response)
            else:
                # Use standard LLM
                answer_chain = self.answer_agent
                try:
                    response = answer_chain.invoke({"question": question, "documents": documents, "images": images})
                    
                    # Handle different response types
                    if isinstance(response, str):
                        answer = response
                    elif hasattr(response, "content"):
                        answer = response.content
                    else:
                        answer = str(response)
                except Exception as e:
                    print(f"LLM error: {e}")
                    answer = f"I encountered an error while generating the answer: {str(e)[:100]}"
            
            # Clean up answer (remove repeated text)
            if answer:
                # Remove excessive repetition
                words = answer.split()
                if len(words) > 200:
                    # If too long, take first 200 words
                    answer = " ".join(words[:200]) + "..."
            
            # Update chat history - prevent exact duplicates
            if not chat_history or chat_history[-1] != ("user", question):
                new_history = chat_history + [("user", question), ("assistant", answer)]
            else:
                # Question already asked, just update answer
                new_history = chat_history[:-1] + [("assistant", answer)]
            
            return {
                "answer": answer,
                "chat_history": new_history
            }
'''

# Apply fixes using regex
# Fix query method
query_pattern = r'(\s+def query\(self, question: str\):.*?)(?=\s+def |\s+class |\Z)'
agents_code = re.sub(query_pattern, new_query_method + '\n', agents_code, flags=re.DOTALL)

# Fix generate_answer method
generate_pattern = r'(\s+def generate_answer\(state\):.*?)(?=\s+def |\s+# Add nodes|\s+workflow\.add_node|\s+workflow\.set_entry_point)'
agents_code = re.sub(generate_pattern, new_generate_answer + '\n', agents_code, flags=re.DOTALL)

# Write fixed file
with open('agents.py', 'w') as f:
    f.write(agents_code)

print("✅ Fixed agents.py successfully!")
print("   ✓ Fixed query() method to extract answer correctly")
print("   ✓ Fixed generate_answer() to prevent infinite loops")
print("   ✓ Added duplicate detection and chat history limits")

