from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
)

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

from langchain_core.output_parsers import StrOutputParser

from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from utils.model_loader import ModelLoader



class AgenticRAG:
    """Agentic RAG pipeline using LangGraph."""

    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        query: str

    def __init__(self):
        self.model_loader = ModelLoader()
        self.llm = self.model_loader.load_llm()
        self.checkpointer = MemorySaver()
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile(checkpointer=self.checkpointer)

    def _history_aware_query(self, state: AgentState):
        """Generates a history-aware query based on the current state."""
        # Implement logic to generate a history-aware query
        # For example, you might concatenate previous messages or use a summarization model
        print('======== Generating history-aware query ====')
        query = state['query']
        if state['messages'] is None or len(state['messages']) == 0:
            return {"query": query}
        history = state['messages'][:-1]
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Rewrite the latest user question into a standalone question.

                        Use the conversation history if needed to resolve references like:
                        - it
                        - they
                        - that product

                        Do not answer the question.

                        Return only the rewritten question.
                    """
                ),
                MessagesPlaceholder("history"),
                (
                    "human", """
                    Given the conversation history, generate a concise query that captures the user's intent.
                    User's Query: {query}
                    History-Aware Query:
                    """
                )
            ]
        )
        chain = prompt | self.llm | StrOutputParser()
        rewritten_query = chain.invoke({"query": query, "history": history})
        return {"query": rewritten_query}


    def _answer_query(self, state: AgentState):
        """Answers the user's query based on the current state."""
        # Implement logic to answer the query
        print('======== Answering query ====')
        query = state['query']

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant that provides answers in a concise manner."
                ),
                MessagesPlaceholder("history"),
                (
                    "human", """
                    Question: {query}
                    Answer:
                    """
                )
            ]
        )

        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"query": query, "history": state["messages"][:-1]})
        return {"messages": [AIMessage(content=response)]}

    def _build_workflow(self):
        """Builds the workflow for the Agentic RAG pipeline."""
        # Define the workflow steps here
        # For example, you might have steps for processing input, querying a database, etc.
        print('======== Building workflow ====')
        workflow = StateGraph(self.AgentState)
        workflow.add_node("HistoryAwareQuery", self._history_aware_query)
        workflow.add_node("AnswerQuery", self._answer_query)

        workflow.add_edge(
            START,
            "HistoryAwareQuery"
        )
        workflow.add_edge(
            "HistoryAwareQuery",
            "AnswerQuery"
        )
        workflow.add_edge("AnswerQuery", END)
        return workflow


    def run(self, query: str,thread_id: str = "default_thread") -> str:
        """Run the workflow for a given query and return the final answer."""
        result = self.app.invoke({
            "messages": [HumanMessage(content=query)],
            "query": query,
        },
        config={"configurable": {"thread_id": thread_id}})
        print("\n--- Workflow Result ---")
        print(result)
        print("\n--- Final Answer ---")
        return result["messages"][-1].content


if __name__ == "__main__":

    rag = AgenticRAG()
    thread = "conversation_1"

    response1 = rag.run("Write a small paragraph about coal washery best practices", thread)
    print(response1)


    response2 = rag.run("What BHP does on this domain", thread)
    print(response2)



