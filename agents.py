"""Multi-agent system using LangGraph."""
import functools
import os
from typing import Annotated, Literal, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
# Try to import agent creation functions
create_openai_functions_agent = None
AgentExecutor = None

try:
    from langchain.agents import create_openai_functions_agent, AgentExecutor
except ImportError:
    try:
        from langchain.agents import AgentExecutor
        from langchain.agents.openai_functions_agent.base import create_openai_functions_agent
    except ImportError:
        pass  # Will use fallback methods
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
import config
from vector_store import MultimodalVectorStore
from huggingface_fallback import HuggingFaceMultimodal, HuggingFaceLLM, HuggingFaceLangChainLLM


class AgentState(TypedDict):
    """State for the multi-agent system."""
    messages: Annotated[list, lambda x, y: x + y]
    next: str
    iteration_count: int  # Track iterations to prevent infinite loops


def create_web_search_tool():
    """Create a web search tool."""
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
        if config.TAVILY_API_KEY:
            return TavilySearchResults(
                max_results=3,
                tavily_api_key=config.TAVILY_API_KEY
            )
    except:
        pass
    
    # Fallback to DuckDuckGo
    try:
        from langchain_community.tools import DuckDuckGoSearchRun
        return DuckDuckGoSearchRun()
    except:
        # If both fail, create a mock tool
        def mock_search(query: str) -> str:
            return f"Web search results for: {query}\n(Web search not configured. Please set TAVILY_API_KEY or install duckduckgo-search)"
        
        return Tool(
            name="web_search",
            description="Search the web for current information",
            func=mock_search
        )


def create_retriever_tools(vector_store: MultimodalVectorStore):
    """Create retriever tools for document search."""
    import config
    # Use smaller k when images might be present to avoid token limits
    retriever = vector_store.get_retriever(k=config.MAX_RETRIEVAL_DOCS)
    
    def retrieve_documents(query: str) -> str:
        """
        Retrieve relevant documents from the vector store.
        Performs both text and image semantic search.
        Uses Hugging Face for image understanding if available.
        """
        import config
        
        # Don't use API-based Hugging Face for image understanding
        # Images will be handled by the local CLIP model in vector_store
        hf_multimodal = None  # Disabled API-based image understanding
        
        # Perform multimodal search (text + images)
        search_results = vector_store.multimodal_search(query, k=config.MAX_RETRIEVAL_DOCS, include_images=True)
        
        docs = search_results["text_results"]
        image_results = search_results["image_results"]
        
        results = []
        max_images_per_response = config.MAX_IMAGES_PER_QUERY  # Limit images to prevent token overflow
        
        # Process text results
        for doc in docs:
            result = {
                "content": doc.page_content,
                "metadata": {
                    "chunk_id": doc.metadata.get("chunk_id"),
                    "page_number": doc.metadata.get("page_number"),
                    "chunk_type": doc.metadata.get("chunk_type"),
                }
            }
            
            if doc.metadata.get("has_image"):
                result["has_image"] = True
                result["image_count"] = doc.metadata.get("image_count", 0)
                
                # Get images from cache (not from metadata)
                chunk_id = doc.metadata.get("chunk_id")
                image_data = vector_store.get_image_for_chunk(chunk_id)
                
                if image_data and image_data.get("images"):
                    images = image_data["images"]
                    # Only include first image to save tokens
                    result["image_base64"] = images[0]
                    result["note"] = f"{len(images)} image(s) available (page {doc.metadata.get('page_number', 'unknown')})"
                    
                    # Note: Image descriptions are handled by CLIP embeddings in vector_store
                    # We don't use API-based image understanding to avoid API errors
                    # Images are already semantically searchable via CLIP embeddings
                    if False:  # Disabled API-based image description
                        pass
                    # Original code (disabled):
                    # if hf_multimodal and hf_multimodal.is_available():
                    #     try:
                    #         image_description = hf_multimodal.describe_image(images[0], f"Describe this image. User query: {query}")
                            if image_description:
                                result["image_description"] = image_description
                                result["note"] += " (Description generated via Hugging Face)"
                        except Exception as e:
                            print(f"Warning: Could not describe image with Hugging Face: {e}")
                else:
                    result["note"] = f"Image available but data not found (page {doc.metadata.get('page_number', 'unknown')})"
            else:
                result["has_image"] = False
                result["image_count"] = 0
            
            results.append(result)
            
            # Limit total images in response to prevent token overflow
            total_images = sum(1 for r in results if r.get("has_image") and r.get("image_base64"))
            if total_images >= max_images_per_response:
                # For remaining docs, just indicate images exist without including them
                break
        
        # Add image search results
        for img_result in image_results[:max_images_per_response]:
            chunk_id = img_result.get("chunk_id")
            # Check if we already have this chunk in text results
            if not any(r.get("metadata", {}).get("chunk_id") == chunk_id for r in results):
                result = {
                    "content": f"Image from page {img_result.get('page_number', 'unknown')} (semantically matched)",
                    "metadata": {
                        "chunk_id": chunk_id,
                        "page_number": img_result.get("page_number"),
                        "chunk_type": "Image",
                    },
                    "has_image": True,
                    "image_base64": img_result.get("image_base64"),
                    "image_count": 1,
                    "note": f"Image found via semantic search (similarity: {1 - img_result.get('distance', 0):.2f})"
                }
                
                # Use Hugging Face to describe image if available
                if hf_multimodal and hf_multimodal.is_available() and img_result.get("image_base64"):
                    try:
                        image_description = hf_multimodal.describe_image(
                            img_result.get("image_base64"), 
                            f"Describe this image in detail. User is asking: {query}"
                        )
                        if image_description:
                            result["image_description"] = image_description
                            result["content"] = f"Image from page {img_result.get('page_number', 'unknown')}: {image_description}"
                            result["note"] += " (Description via Hugging Face)"
                    except Exception as e:
                        print(f"Warning: Could not describe image with Hugging Face: {e}")
                
                results.append(result)
        
        # If we have more docs, add them without images
        remaining_docs = docs[len(results):]
        for doc in remaining_docs:
            result = {
                "content": doc.page_content,
                "metadata": {
                    "chunk_id": doc.metadata.get("chunk_id"),
                    "page_number": doc.metadata.get("page_number"),
                    "chunk_type": doc.metadata.get("chunk_type"),
                },
                "has_image": doc.metadata.get("has_image", False),
                "image_count": doc.metadata.get("image_count", 0) if doc.metadata.get("has_image") else 0,
            }
            if result["has_image"]:
                result["note"] = f"Image available but not included to save tokens (page {doc.metadata.get('page_number', 'unknown')})"
            results.append(result)
        
        return str(results)
    
    return [
        Tool(
            name="retrieve_documents",
            description="Retrieve relevant documents from the PDF knowledge base. Use this when the user asks about information that might be in the uploaded PDFs.",
            func=retrieve_documents
        )
    ]


def create_agent(llm, tools, system_prompt: str):
    """Create an agent with the given tools."""
    # Check if we're using Hugging Face LLM - skip OpenAI-specific methods
    is_huggingface = hasattr(llm, '_llm_type') and getattr(llm, '_llm_type', None) == "huggingface"
    is_huggingface = is_huggingface or (hasattr(llm, 'model_name') and 'huggingface' in str(type(llm)).lower())
    is_huggingface = is_huggingface or not hasattr(llm, 'bind_functions')
    
    # Skip OpenAI-specific methods for Hugging Face
    if not is_huggingface and create_openai_functions_agent is not None and AgentExecutor is not None:
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="messages"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
            
            agent = create_openai_functions_agent(llm, tools, prompt)
            return AgentExecutor(agent=agent, tools=tools, verbose=True)
        except Exception as e:
            print(f"Warning: create_openai_functions_agent failed: {e}, using fallback")
    
    # Use create_react_agent for Hugging Face or as fallback (works with any LLM)
    from langgraph.prebuilt import create_react_agent
    try:
        # Check if LLM is Hugging Face - create_react_agent should work but let's be explicit
        if is_huggingface:
            print("Creating agent with Hugging Face LLM using create_react_agent...")
        agent_graph = create_react_agent(llm, tools)
        # Store system prompt as attribute for later use in agent_node
        agent_graph.system_prompt = system_prompt
        print(f"✓ Created agent using create_react_agent (compatible with Hugging Face)")
        return agent_graph
    except Exception as e:
        error_msg = str(e).lower()
        # Check if error is from OpenAI (shouldn't happen with Hugging Face)
        if "openai" in error_msg or "429" in error_msg or "quota" in error_msg:
            print(f"⚠️ ERROR: create_react_agent tried to use OpenAI with Hugging Face LLM! Error: {e}")
            raise ValueError(f"create_react_agent attempted to use OpenAI when Hugging Face LLM was provided. This is a bug. Error: {e}")
        print(f"Warning: create_react_agent failed: {e}, using simple agent")
        # If create_react_agent fails, create a simple wrapper
        class SimpleAgent:
            def __init__(self, llm, tools, system_prompt):
                self.llm = llm
                self.tools = tools
                self.system_prompt = system_prompt
            
            def invoke(self, state):
                from langchain_core.messages import SystemMessage, HumanMessage
                messages = state.get("messages", [])
                # Add system prompt
                if self.system_prompt:
                    messages = [SystemMessage(content=self.system_prompt)] + messages
                
                # Simple tool execution
                last_message = messages[-1].content if messages else ""
                result = f"Agent response for: {last_message}"
                return {"output": result, "messages": messages}
        
        return SimpleAgent(llm, tools, system_prompt)


def agent_node(state: AgentState, agent, name: str):
    """Execute an agent node."""
    try:
        # For AgentExecutor
        result = agent.invoke(state)
        if isinstance(result, dict) and "output" in result:
            output = result["output"]
            # Ensure the output is clear and complete
            if not output or len(output.strip()) < 10:
                output = f"I processed the query but didn't find relevant information."
            return {"messages": [AIMessage(content=output)]}
        else:
            return {"messages": [AIMessage(content=str(result))]}
    except (AttributeError, TypeError) as e:
        # For create_react_agent or graph-based agents
        # Add system message if available
        messages = state.get("messages", [])
        if hasattr(agent, "system_prompt") and agent.system_prompt:
            from langchain_core.messages import SystemMessage
            messages = [SystemMessage(content=agent.system_prompt)] + messages
        
        # Invoke with messages
        try:
            if hasattr(agent, "invoke"):
                result = agent.invoke({"messages": messages})
            else:
                result = agent.invoke(state)
            
            if isinstance(result, dict) and "messages" in result:
                # Extract the last message content
                last_msg = result["messages"][-1] if result["messages"] else None
                if last_msg and hasattr(last_msg, "content"):
                    return {"messages": [AIMessage(content=last_msg.content)]}
                return result
            else:
                return {"messages": [AIMessage(content=str(result))]}
        except Exception as e:
            # If agent fails, return an error message
            return {"messages": [AIMessage(content=f"Agent {name} encountered an error: {str(e)}")]}


class MultiAgentRAG:
    """Multi-agent RAG system using LangGraph."""
    
    def __init__(self, vector_store: MultimodalVectorStore, use_huggingface_primary: bool = None):
        """
        Initialize the multi-agent RAG system.
        
        Args:
            vector_store: Initialized vector store instance
            use_huggingface_primary: If True, use Hugging Face as primary LLM (default: True if HUGGINGFACE_API_KEY set)
        """
        self.vector_store = vector_store
        
        # Determine which LLM to use as primary
        if use_huggingface_primary is None:
            # Auto-detect: use Hugging Face if API key is available
            use_huggingface_primary = bool(config.HUGGINGFACE_API_KEY)
        
        # Initialize primary LLM (Hugging Face) - use LOCAL models instead of API
        if use_huggingface_primary:
            print("Using Hugging Face as primary LLM (local model)")
            try:
                # Use local Hugging Face model via transformers (no API needed)
                from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
                from langchain_community.llms import HuggingFacePipeline
                import torch
                
                model_name = config.HUGGINGFACE_LLM_MODEL
                print(f"Loading local Hugging Face model: {model_name}")
                
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"Using device: {device}")
                
                # Load model and tokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(model_name)
                
                # Create pipeline
                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=500,
                    device=0 if device == "cuda" else -1,
                    temperature=config.TEMPERATURE,
                )
                
                # Wrap in LangChain compatible interface
                self.llm = HuggingFacePipeline(pipeline=pipe)
                self.primary_llm_type = "huggingface"
                
                # Keep OpenAI as fallback (lazy initialization - only create when needed)
                self._openai_api_key = config.OPENAI_API_KEY
                self._openai_config = {
                    "model": config.LLM_MODEL,
                    "temperature": config.TEMPERATURE,
                    "max_retries": config.MAX_RETRIES,
                    "request_timeout": 60
                }
                self.fallback_llm = None  # Will be created lazily if needed
                print("✓ Hugging Face primary LLM initialized (local model, no API needed)")
            except Exception as e:
                print(f"❌ Error: Failed to initialize local Hugging Face LLM: {e}")
                import traceback
                traceback.print_exc()
                # Don't fall back to API - local models are more reliable
                # Instead, provide helpful error message
                error_msg = f"Failed to load local Hugging Face model '{model_name}'. "
                error_msg += "This might be due to:\n"
                error_msg += "1. Insufficient memory (try a smaller model like 'gpt2')\n"
                error_msg += "2. Network issues downloading the model\n"
                error_msg += "3. Missing dependencies (ensure transformers and torch are installed)\n\n"
                error_msg += f"Full error: {str(e)}"
                raise ValueError(error_msg)
                # Try OpenAI as fallback, but handle quota errors gracefully
                if config.OPENAI_API_KEY:
                    try:
                        self.llm = ChatOpenAI(
                            model=config.LLM_MODEL,
                            temperature=config.TEMPERATURE,
                            api_key=config.OPENAI_API_KEY,
                            max_retries=0,  # Don't retry during initialization
                            request_timeout=5
                        )
                        self.primary_llm_type = "openai"
                        self.fallback_llm = None
                        print("Falling back to OpenAI as primary")
                    except Exception as openai_error:
                        error_str = str(openai_error).lower()
                        if "quota" in error_str or "429" in error_str:
                            raise ValueError("Both Hugging Face and OpenAI are unavailable. Hugging Face failed to initialize and OpenAI quota is exceeded. Please check your Hugging Face API key or wait for OpenAI quota to reset.")
                        else:
                            raise ValueError(f"Failed to initialize both Hugging Face and OpenAI: {e}")
                else:
                    raise ValueError(f"Failed to initialize Hugging Face LLM and no OpenAI API key available: {e}")
        else:
            # Use OpenAI as primary
            print("Using OpenAI as primary LLM")
            try:
                self.llm = ChatOpenAI(
                    model=config.LLM_MODEL,
                    temperature=config.TEMPERATURE,
                    api_key=config.OPENAI_API_KEY,
                    max_retries=config.MAX_RETRIES,
                    request_timeout=60
                )
                self.primary_llm_type = "openai"
            except Exception as e:
                error_str = str(e).lower()
                if "quota" in error_str or "429" in error_str:
                    # If OpenAI quota exceeded, try local Hugging Face model instead
                    print("OpenAI quota exceeded, switching to local Hugging Face model...")
                    try:
                        # Use local Hugging Face model (same as primary initialization)
                        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
                        from langchain_community.llms import HuggingFacePipeline
                        import torch
                        
                        model_name = config.HUGGINGFACE_LLM_MODEL
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        
                        tokenizer = AutoTokenizer.from_pretrained(model_name)
                        model = AutoModelForCausalLM.from_pretrained(model_name)
                        pipe = pipeline(
                            "text-generation",
                            model=model,
                            tokenizer=tokenizer,
                            max_new_tokens=500,
                            device=0 if device == "cuda" else -1,
                            temperature=config.TEMPERATURE,
                        )
                        self.llm = HuggingFacePipeline(pipeline=pipe)
                        self.primary_llm_type = "huggingface"
                        self.fallback_llm = None
                        print("✓ Using local Hugging Face model (OpenAI quota exceeded)")
                    except Exception as hf_error:
                        raise ValueError(f"OpenAI quota exceeded and local Hugging Face model initialization failed: {hf_error}")
                else:
                    raise e
            
            # Don't use API-based Hugging Face as fallback - we use local models
            # If local model fails, we'll handle it in error handling
            self.fallback_llm = None
            else:
                self.fallback_llm = None
        
        # Create tools
        self.retriever_tools = create_retriever_tools(vector_store)
        self.web_search_tool = create_web_search_tool()
        
        # Create agents
        self.retriever_agent = create_agent(
            self.llm,
            self.retriever_tools,
            "You are a helpful assistant that retrieves information from PDF documents. "
            "When you retrieve documents, provide clear and concise summaries of the relevant information."
        )
        
        self.web_search_agent = create_agent(
            self.llm,
            [self.web_search_tool],
            "You are a helpful assistant that searches the web for current information. "
            "Provide accurate and up-to-date information from web search results."
        )
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build the LangGraph workflow."""
        members = ["retriever", "web_search"]
        
        # System prompt for supervisor
        system_prompt = """You are a supervisor tasked with managing a conversation between the following workers: {members}.
        Given the following user question, respond with the worker to act next. Each worker will perform a task and respond with their results and status.
        
        IMPORTANT RULES:
        1. If the question is about information in the uploaded PDF documents, use "retriever"
        2. If the question requires current/up-to-date information not in PDFs, use "web_search"
        3. After an agent responds, if the answer is complete and satisfactory, respond with "FINISH"
        4. If you need more information, you can route to another agent, but avoid loops
        5. Always respond with FINISH after getting a satisfactory answer
        
        Available workers:
        - retriever: Retrieves information from uploaded PDF documents. Use this when the question relates to the PDF content.
        - web_search: Searches the web for current information. Use this when the question requires up-to-date information not in the PDFs.
        
        Begin!"""
        
        # Supervisor prompt
        options = ["FINISH"] + members
        options_str = ", ".join(options)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the conversation above, who should act next? "
                "Or should we FINISH? You MUST respond with ONLY one of these options: {options}. "
                "Respond in JSON format: {{\"next\": \"<option>\"}}",
            ),
        ]).partial(options=options_str, members=", ".join(members))
        
        # Use function calling (more compatible with older models)
        function_def = {
            "name": "route",
            "description": "Select the next role to route to.",
            "parameters": {
                "type": "object",
                "properties": {
                    "next": {
                        "type": "string",
                        "enum": options,
                        "description": "The next agent to route to or FINISH to end"
                    }
                },
                "required": ["next"]
            }
        }
        
        # Check if we're using Hugging Face - if so, skip OpenAI-specific methods
        use_simple_parsing = self.primary_llm_type == "huggingface" or not hasattr(self.llm, 'bind_functions')
        
        supervisor_chain = None
        
        if not use_simple_parsing:
            try:
                # Try with bind_functions (works with OpenAI models)
                supervisor_chain = (
                    prompt
                    | self.llm.bind_functions(functions=[function_def], function_call={"name": "route"})
                    | JsonOutputFunctionsParser()
                )
            except Exception as e:
                print(f"bind_functions failed: {e}, trying fallback...")
                use_simple_parsing = True  # Force simple parsing
        
        if use_simple_parsing and supervisor_chain is None:
            # For Hugging Face or when OpenAI methods fail, try structured output first
            try:
                from pydantic import BaseModel, Field
                from typing import Literal
                
                class RouteDecision(BaseModel):
                    """Route decision model."""
                    next: Literal["retriever", "web_search", "FINISH"] = Field(
                        description="The next agent to route to or FINISH to end"
                    )
                
                if hasattr(self.llm, 'with_structured_output'):
                    supervisor_chain = (
                        prompt
                        | self.llm.with_structured_output(RouteDecision, method="function_calling")
                    )
            except Exception as e2:
                print(f"Structured output also failed: {e2}, using simple text parsing...")
                supervisor_chain = None
        
        # Define supervisor_node - use chain if available, otherwise simple parsing
        if supervisor_chain is not None:
            def supervisor_node(state: AgentState):
                # Check iteration count to prevent infinite loops
                iteration_count = state.get("iteration_count", 0)
                if iteration_count >= config.MAX_ITERATIONS - 1:
                    return {"next": "FINISH", "iteration_count": iteration_count + 1}
                
                result = supervisor_chain.invoke(state)
                next_agent = "FINISH"
                if isinstance(result, dict):
                    next_agent = result.get("next", "FINISH")
                elif hasattr(result, "next"):
                    next_agent = result.next
                
                # Safety check: if we've been routing too much, force FINISH
                if iteration_count >= 10 and next_agent != "FINISH":
                    next_agent = "FINISH"
                
                return {"next": next_agent, "iteration_count": iteration_count + 1}
        else:
            # Simple text parsing (works with any LLM, including Hugging Face)
            def supervisor_node(state: AgentState):
                # Check iteration count to prevent infinite loops
                iteration_count = state.get("iteration_count", 0)
                if iteration_count >= config.MAX_ITERATIONS - 1:
                    return {"next": "FINISH", "iteration_count": iteration_count + 1}
                
                messages = state.get("messages", [])
                formatted_messages = prompt.format_messages(**{"messages": messages})
                response = self.llm.invoke(formatted_messages)
                content = response.content if hasattr(response, "content") else str(response)
                
                # Try to parse JSON from response
                import json
                import re
                json_match = re.search(r'\{[^}]+\}', content)
                if json_match:
                    try:
                        result = json.loads(json_match.group())
                        next_agent = result.get("next", "FINISH")
                    except:
                        # Fallback: check if any option is mentioned
                        next_agent = "FINISH"
                        for option in options:
                            if option.lower() in content.lower():
                                next_agent = option
                                break
                else:
                    # Check if any option is mentioned in text
                    next_agent = "FINISH"
                    for option in options:
                        if option.lower() in content.lower():
                            next_agent = option
                            break
                
                # Safety check: if we've been routing too much, force FINISH
                if iteration_count >= 10 and next_agent != "FINISH":
                    next_agent = "FINISH"
                
                return {"next": next_agent, "iteration_count": iteration_count + 1}
        
        # Create graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        retriever_node = functools.partial(
            agent_node,
            agent=self.retriever_agent,
            name="retriever"
        )
        workflow.add_node("retriever", retriever_node)
        
        search_node = functools.partial(
            agent_node,
            agent=self.web_search_agent,
            name="web_search"
        )
        workflow.add_node("web_search", search_node)
        
        # supervisor_node is defined in the try/except block above
        # If it wasn't defined (fallback case), define it here
        if 'supervisor_node' not in locals():
            def supervisor_node(state: AgentState):
                # Check iteration count to prevent infinite loops
                iteration_count = state.get("iteration_count", 0)
                if iteration_count >= config.MAX_ITERATIONS - 1:
                    return {"next": "FINISH", "iteration_count": iteration_count + 1}
                
                result = supervisor_chain.invoke(state)
                next_agent = "FINISH"
                if isinstance(result, dict):
                    next_agent = result.get("next", "FINISH")
                elif hasattr(result, "next"):
                    next_agent = result.next
                
                # Safety check: if we've been routing too much, force FINISH
                if iteration_count >= 10 and next_agent != "FINISH":
                    next_agent = "FINISH"
                
                return {"next": next_agent, "iteration_count": iteration_count + 1}
        
        workflow.add_node("supervisor", supervisor_node)
        
        # Add edges
        for member in members:
            workflow.add_edge(member, "supervisor")
        
        # Conditional edges from supervisor
        conditional_map = {k: k for k in members}
        conditional_map["FINISH"] = END
        workflow.add_conditional_edges(
            "supervisor",
            lambda x: x["next"],
            conditional_map
        )
        
        # Set entry point
        workflow.set_entry_point("supervisor")
        
        return workflow.compile()
    
    def query(self, question: str) -> str:
        """
        Query the multi-agent RAG system.
        
        Args:
            question: User question
            
        Returns:
            Answer from the system
        """
        import time
        import random
        
        graph_config = {
            "recursion_limit": config.MAX_ITERATIONS
        }
        
        # Initialize state with iteration count
        initial_state = {
            "messages": [HumanMessage(content=question)],
            "next": "",
            "iteration_count": 0
        }
        
        max_retries = 3
        retry_delay = 2  # Start with 2 seconds
        
        for attempt in range(max_retries):
            try:
                result = self.graph.invoke(initial_state, config=graph_config)
                
                # Extract the final answer
                messages = result.get("messages", [])
                if messages:
                    # Get the last AI message
                    for msg in reversed(messages):
                        if hasattr(msg, "content") and msg.content:
                            return msg.content
                    return str(messages[-1])
                return "No response generated."
                
            except Exception as e:
                error_str = str(e).lower()
                
                # If using Hugging Face as primary and it fails, DO NOT try OpenAI fallback
                # OpenAI quota is exceeded, so we should NEVER attempt to use it
                if self.primary_llm_type == "huggingface":
                    # Check if error mentions OpenAI (shouldn't happen, but indicates a bug)
                    if "openai" in error_str or "429" in error_str or "quota" in error_str:
                        print(f"⚠️ ERROR: OpenAI error detected when using Hugging Face primary! This shouldn't happen.")
                        print(f"Full error: {str(e)}")
                        return f"Error: The system tried to use OpenAI even though Hugging Face is primary. This is a bug. Error details: {str(e)[:300]}. Please check that all components are using Hugging Face."
                    
                    # Return Hugging Face error without trying OpenAI fallback
                    error_str_lower = str(e).lower()
                    
                    # Check if this is an API error (shouldn't happen with local models)
                    if "api" in error_str_lower or "endpoint" in error_str_lower or "hugging face api" in error_str_lower:
                        print(f"⚠️ ERROR: API error detected when using local Hugging Face model!")
                        print(f"This indicates the system is incorrectly trying to use the API.")
                        print(f"Full error: {str(e)}")
                        return f"ERROR: The system tried to use Hugging Face API even though local models are configured. This is a bug. Please re-run Step 6 to reinitialize the system with local models. Error: {str(e)[:300]}"
                    
                    # This is a local model error
                    print(f"⚠️ Hugging Face error: {str(e)[:200]}")
                    error_msg = f"Error with local Hugging Face model: {str(e)[:200]}. "
                    error_msg += "This is a local model error (not an API error). "
                    error_msg += "Please check that the model loaded correctly or try reinitializing the system (re-run Step 6)."
                    return error_msg
                
                # Handle rate limit/quota errors - ONLY if OpenAI is primary
                # If Hugging Face is primary, these errors shouldn't happen (they indicate a bug)
                if self.primary_llm_type == "huggingface":
                    # If we see OpenAI errors when using Hugging Face, something is wrong
                    if "openai" in error_str or "429" in error_str or "quota" in error_str:
                        print(f"⚠️ CRITICAL: OpenAI error detected when Hugging Face is primary!")
                        print(f"Full error: {str(e)}")
                        return f"ERROR: The system tried to use OpenAI even though Hugging Face is primary. This indicates a configuration bug. Error: {str(e)[:300]}. Please ensure all components are using Hugging Face."
                    
                    # For other errors with Hugging Face, return the error directly
                    return f"Hugging Face error: {str(e)[:300]}. Please check your Hugging Face API key or configuration."
                
                # Handle rate limit/quota errors - use Hugging Face fallback if OpenAI is primary
                if ("rate limit" in error_str or "429" in error_str or "tpm" in error_str or 
                    "quota" in error_str or "exceeded" in error_str):
                    # Extract wait time from error message
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    if "try again in" in error_str or "retry after" in error_str:
                        try:
                            import re
                            # Try to extract wait time (could be in seconds or milliseconds)
                            wait_match = re.search(r'(?:try again in|retry after)\s*(\d+)\s*(?:ms|seconds?|s)?', error_str, re.IGNORECASE)
                            if wait_match:
                                extracted_wait = int(wait_match.group(1))
                                # If it's in milliseconds, convert to seconds
                                if "ms" in error_str.lower():
                                    wait_time = extracted_wait / 1000 + 5  # Add 5 second buffer
                                else:
                                    wait_time = extracted_wait + 5  # Add 5 second buffer
                        except:
                            pass
                    
                    # Don't use API-based Hugging Face fallback - we use local models
                    # This code path should only be used if OpenAI is primary
                    if attempt == 0 and self.primary_llm_type == "openai":
                        # Skip API-based fallback - not needed when using local models
                        pass
                        
                        if self.fallback_llm:
                            try:
                                print("OpenAI quota/rate limit exceeded, switching to Hugging Face...")
                                # Temporarily switch to Hugging Face
                                original_llm = self.llm
                                self.llm = self.fallback_llm
                                # Also update agents
                                self.retriever_agent = create_agent(
                                    self.llm,
                                    self.retriever_tools,
                                    "You are a helpful assistant that retrieves information from PDF documents. "
                                    "When you retrieve documents, provide clear and concise summaries of the relevant information."
                                )
                                self.web_search_agent = create_agent(
                                    self.llm,
                                    [self.web_search_tool],
                                    "You are a helpful assistant that searches the web for current information. "
                                    "Provide accurate and up-to-date information from web search results."
                                )
                                # Rebuild graph with new LLM
                                self.graph = self._build_graph()
                                # Retry query
                                result = self.graph.invoke(initial_state, config=graph_config)
                                messages = result.get("messages", [])
                                if messages:
                                    for msg in reversed(messages):
                                        if hasattr(msg, "content") and msg.content:
                                            return f"[Using Hugging Face fallback due to OpenAI quota/rate limit]\n\n{msg.content}"
                                # Restore original LLM
                                self.llm = original_llm
                            except Exception as hf_error:
                                print(f"Hugging Face fallback failed: {hf_error}")
                                # Restore original LLM if it was changed
                                if hasattr(self, 'original_llm'):
                                    self.llm = self.original_llm
                    
                    # Retry with delay
                    if attempt < max_retries - 1:
                        wait_time = min(wait_time, 120)  # Cap at 120 seconds
                        print(f"Rate limit hit. Waiting {wait_time:.1f} seconds before retry {attempt + 1}/{max_retries}...")
                        time.sleep(wait_time)
                        continue
                    else:
                        # Only return OpenAI error message if OpenAI is actually primary
                        if self.primary_llm_type == "openai":
                            return f"Rate limit error: OpenAI API rate limit reached after {max_retries} attempts. Please wait a few minutes and try again. Error: {str(e)[:200]}"
                        else:
                            # This shouldn't happen - indicates a bug
                            return f"ERROR: Rate limit error detected but Hugging Face is primary. This indicates OpenAI is being called somewhere it shouldn't be. Error: {str(e)[:300]}"
                
                # Handle recursion limit
                if "recursion limit" in error_str:
                    try:
                        # Try with a simpler direct query
                        if self.vector_store and hasattr(self.vector_store, "get_retriever"):
                            retriever = self.vector_store.get_retriever(k=3)
                            docs = retriever.get_relevant_documents(question)
                            if docs:
                                context = "\n\n".join([doc.page_content for doc in docs[:3]])
                                
                                # Try OpenAI first, fallback to Hugging Face if rate limited
                                try:
                                    response = self.llm.invoke([
                                        HumanMessage(content=f"Based on this context: {context}\n\nAnswer this question: {question}")
                                    ])
                                    return response.content if hasattr(response, "content") else str(response)
                                except Exception as llm_error:
                                    # Don't use API-based Hugging Face fallback - we're already using local models
                                    # If this is called, it means the local model failed, so just return the error
                                    error_msg = f"Error with local Hugging Face model: {str(llm_error)[:200]}"
                                    print(f"⚠️ {error_msg}")
                                    return error_msg
                    except:
                        pass
                
                # For other errors, return error message
                if attempt == max_retries - 1:
                    return f"Error processing query: {str(e)[:500]}"
                else:
                    # Retry with exponential backoff
                    time.sleep(retry_delay * (2 ** attempt))
                    continue
        
        return "Error: Maximum retries exceeded."

