from langchain.chains import ConversationalRetrievalChain, load_summarize_chain
from langchain.agents import AgentExecutor, Tool, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

from .prompts import SUMMARIZE_PROMPT

def create_rag_chain(vectorstore, llm, prompt_template, memory: ConversationBufferMemory):
    """
    Creates a RAG (Retrieval-Augmented Generation) chain for question-answering.
    This chain retrieves relevant documents from the vector store and uses the LLM to answer questions based on those documents.
    Memory is used to maintain the context of the conversation.
    """
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory, 
        combine_docs_chain_kwargs={"prompt": prompt_template},
    )
    return chain

def create_summarization_chain(llm, docs):
    """
    Creates a summarization chain for the entire document.
    """
    # Summarize the first 20 chunks for a fast, high-level overview.
    chain = load_summarize_chain(llm, chain_type="stuff", prompt=SUMMARIZE_PROMPT)
    
    def summarizer(input_str: str):
        # The result from the chain is a dictionary, so we extract the 'output_text'.
        result = chain.invoke({"input_documents": docs[:20], "question": input_str})
        return result.get('output_text', "I couldn't generate a summary.")
        
    return summarizer

def create_router_agent(general_rag_chain, financial_rag_chain, summarization_func, llm: ChatOpenAI):
    """
    Creates a router agent that decides which tool (specialized chain) to use.
    """
    tools = [
        Tool(
            name="General_Information_Search",
            # The lambda function is now correct because the chain it calls is stateful.
            func=lambda agent_input: general_rag_chain.invoke({"question": agent_input}).get('answer', 'No answer found for the general question.'),
            description="""
            Use this tool for questions about qualitative information, explanations, strategies, policies, and other narrative content.
            """
        ),
        Tool(
            name="Financial_Data_Search",
            func=lambda agent_input: financial_rag_chain.invoke({"question": agent_input}).get('answer', 'No answer found for the general question.'),
            description="""
            Use this tool for questions requiring specific financial metrics, figures, or data points like revenue, profit, or share prices.
            """
        ),
        Tool(
            name="Document_Summarizer",
            func=summarization_func,
            description="""
            Use this tool when the user asks for a summary, overview, or the key takeaways of a topic.
            The input to this tool should be the topic the user wants summarized.
            """
        ),
    ]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that routes a user's question to the correct tool."),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm, tools, prompt)
    
    router_agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        handle_parsing_errors=True
    )
    
    return router_agent_executor
