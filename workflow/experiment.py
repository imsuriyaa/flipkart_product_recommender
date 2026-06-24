from typing import Annotated, Sequence, TypedDict, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from prompt_library.prompts import PROMPT_REGISTRY, PromptType
from retriever.retrieval import Retriever
from utils.model_loader import ModelLoader
from langgraph.checkpoint.memory import MemorySaver
import asyncio
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_classic.chains import create_history_aware_retriever
from langchain_classic import hub


class AgenticRAG:
    """Agentic RAG pipeline using LangGraph."""

    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        query: str                 # Original user query
        standalone_query: str      # Query after history rewriting
        documents: list
        context: str
        retry_count: int
        route: Literal["retriever", "chat"] | None

    def __init__(self):
        self.retriever_obj = Retriever()
        self.model_loader = ModelLoader()
        self.llm = self.model_loader.load_llm()
        self.checkpointer = MemorySaver()
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile(checkpointer=self.checkpointer)

    # ---------- Helpers ----------
    def _format_docs(self, docs) -> str:
        if not docs:
            return "No relevant documents found."
        formatted_chunks = []
        for d in docs:
            meta = d.metadata or {}
            formatted = (
                f"Title: {meta.get('product_title', 'N/A')}\n"
                f"Price: {meta.get('price', 'N/A')}\n"
                f"Rating: {meta.get('rating', 'N/A')}\n"
                f"Reviews:\n{d.page_content.strip()}"
            )
            formatted_chunks.append(formatted)
        return "\n\n---\n\n".join(formatted_chunks)

    # def _get_history(self,session_id:str) -> BaseChatMessageHistory:
    #     if session_id not in self.history_store:
    #         self.history_store[session_id] = ChatMessageHistory()
        # return self.history_store[session_id]

    # ---------- Nodes ----------
    def _ai_assistant(self, state: AgentState):
        print("--- CALL ASSISTANT ---")
        retry_count = state.get("retry_count", 0)
        if retry_count > 2:
            return {
                "messages": [AIMessage(content="I'm sorry, I couldn't find relevant information to answer your question.")],
                "route": END
            }
        messages = state["messages"]
        last_message = messages[-1].content

        determine_prompt = PROMPT_REGISTRY[PromptType.DETERMINISTIC_BOT].template
        determine_chain = ChatPromptTemplate.from_template(determine_prompt) | self.llm | StrOutputParser()
        relevance = determine_chain.invoke({"question": last_message, "docs": state.get("context", "")})

        if relevance.strip().lower() == "yes":
            return {
                "query": last_message,
                "route": "retriever"
            }
        
        prompt = ChatPromptTemplate.from_template(
            "You are a helpful assistant. Answer the user directly.\n\nQuestion: {question}\nAnswer:"
        )
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"question": last_message})
        return {
            "query": last_message,
            'route': END,
            "messages": [AIMessage(content=response)]
        }

    def _history_rewrite(self, state: AgentState):
        print("--- REWRITE QUESTION WITH HISTORY ---")

        current_question = state["messages"][-1].content

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    PROMPT_REGISTRY[
                        PromptType.REWRITE_QUESTION_WITH_HISTORY
                    ].template
                ),
                MessagesPlaceholder("history"),
                ("human", "{question}")
            ]
        )

        chain = prompt | self.llm | StrOutputParser()

        standalone = chain.invoke(
            {
                "history": state["messages"][:-1],
                "question": current_question,
            }
        )

        return {
            "standalone_query": standalone
        }



    def _vector_retriever(self, state: AgentState):
        
        print("--- RETRIEVER ---")
        query = state["standalone_query"]
        retriever = self.retriever_obj.load_retriever()

        # rephrase_prompt = ChatPromptTemplate.from_template(
        #     PROMPT_REGISTRY[PromptType.REWRITE_QUESTION_WITH_HISTORY].template
        # )

        # history_retriever = create_history_aware_retriever(
        #     self.llm,
        #     retriever,
        #     rephrase_prompt,
        # )

        # docs = history_retriever.invoke({
        #     "chat_history": state["messages"][:-1],
        #     "input": state["messages"][-1].content,
        # })

        docs = retriever.invoke(query)
        context = self._format_docs(docs)
        return {
            "documents": docs,
            "context": context
        }


    def _grade_documents(self, state: AgentState) -> Literal["generator", "rewriter"]:
        print("--- GRADER ---")
        print("Current query:", state)
        question = state["query"]
        docs = state["context"]

        prompt = PromptTemplate(
            template="""You are a grader. Question: {question}\nDocs: {docs}\n
            Are docs relevant to the question? Answer yes or no.""",
            input_variables=["question", "docs"],
        )
        chain = prompt | self.llm | StrOutputParser()
        score = chain.invoke({"question": question, "docs": docs})
        return "generator" if "yes" in score.lower() else "rewriter"


    def _generate(self, state: AgentState):
        print("--- GENERATE ---")
        print("Current query:", state)
        question = state["standalone_query"]
        docs = state["context"]
        prompt = ChatPromptTemplate.from_template(
            PROMPT_REGISTRY[PromptType.PRODUCT_BOT].template
        )
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"history": state["messages"], "context": docs, "question": question})
        return {"messages": [AIMessage(content=response)]}


    def _rewrite(self, state: AgentState):
        print("--- REWRITE ---")
        print("Current query:", state)
        question = state["query"]
        new_q = self.llm.invoke(
            [HumanMessage(content=f"Rewrite the query to be clearer: {question}")]
        )
        return {
            "retry_count": state.get("retry_count", 0) + 1,
            "messages": [HumanMessage(content=new_q.content)], 
            "query": new_q.content,
            "standalone_query": new_q.content,
        }


    # ---------- Build Workflow ----------
    def _build_workflow(self):
        workflow = StateGraph(self.AgentState)
        workflow.add_node("Assistant", self._ai_assistant)
        workflow.add_node("HistoryRewrite", self._history_rewrite)
        workflow.add_node("Retriever", self._vector_retriever)
        workflow.add_node("Generator", self._generate)
        workflow.add_node("Rewriter", self._rewrite)

        workflow.add_edge(START, "Assistant")
        # workflow.add_conditional_edges(
        #     "Assistant",
        #     lambda state: "Retriever" if state["route"] == "retriever" else END,
        #     {"Retriever": "Retriever", "generator": "Generator",  END: END},
        # )
        workflow.add_conditional_edges(
            "Assistant",
            lambda state: state["route"],
            {"retriever": "HistoryRewrite", END: END},
        )
        workflow.add_edge("HistoryRewrite", "Retriever")
        workflow.add_conditional_edges(
            "Retriever",
            self._grade_documents,
            {"generator": "Generator", "rewriter": "Rewriter"},
        )
        workflow.add_edge("Generator", END)
        workflow.add_edge("Rewriter", "Assistant")
        return workflow

    # ---------- Public Run ----------
    def run(self, query: str,thread_id: str = "default_thread") -> str:
        """Run the workflow for a given query and return the final answer."""
        result = self.app.invoke({
            "messages": [HumanMessage(content=query)],
            "query": query,
            "retry_count": 0
        },
        config={"configurable": {"thread_id": thread_id}})
        print("\n--- Workflow Result ---")
        print(result)
        print("\n--- Final Answer ---")
        return result["messages"][-1].content
    
        # function call with be asscoiate
        # you will get some score
        # put condition behalf on that score
        # if relevany>0.75
            #return
        #else:
            #contine


if __name__ == "__main__":
    
    
    rag_agent = AgenticRAG()
    answer = rag_agent.run("iphone 16 pro max prices?")
    print("\nFinal Answer:\n", answer)
    
    
    # retrieved_contexts,response = invoke_chain(user_query)
    
    # #this is not an actual output this have been written to test the pipeline
    # #response="iphone 16 plus, iphone 16, iphone 15 are best phones under 1,00,000 INR."
    
    # context_score = evaluate_context_precision(user_query,response,retrieved_contexts)
    # relevancy_score = evaluate_response_relevancy(user_query,response,retrieved_contexts)
    
    # print("\n--- Evaluation Metrics ---")
    # print("Context Precision Score:", context_score)
    # print("Response Relevancy Score:", relevancy_score)d