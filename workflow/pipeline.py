from operator import itemgetter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
# RunnableWithMessageHistory - handles updates of the history of the chat
from langchain_core.runnables.history import RunnableWithMessageHistory
# ChatMessageHistory - maintains the history of the chat (kind of datatype)
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from utils.model_loader import ModelLoader

class RAGChainBuilder:
    def __init__(self,vector_store):
        self.vector_store=vector_store
        self.model = ModelLoader().load_llm()
        self.history_store={}

    def _get_history(self,session_id:str) -> BaseChatMessageHistory:
        if session_id not in self.history_store:
            self.history_store[session_id] = ChatMessageHistory()
        return self.history_store[session_id]
    
    def build_chain(self):
        retriever = self.vector_store.as_retriever(search_kwargs={"k":3})

        context_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given the chat history and user question, rewrite it as a standalone question."),
            MessagesPlaceholder(variable_name="chat_history"), 
            ("human", "{input}")  
        ])

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """You're an e-commerce bot answering product-related queries using reviews and titles.
                          Stick to context. Be concise and helpful.\n\nCONTEXT:\n{context}\n\nQUESTION: {input}"""),
            MessagesPlaceholder(variable_name="chat_history"), 
            ("human", "{input}")  
        ])

        history_aware_retriever = context_prompt | self.model | StrOutputParser()

        question_answer_chain = ({
                    "context": itemgetter("input") | retriever,
                    "input": itemgetter("input"),
                    "chat_history": itemgetter("chat_history")
                }
                | qa_prompt
                | self.model
                | StrOutputParser()
        )

        def rewrite_and_pass(input_dict):
            rewritten = history_aware_retriever.invoke(input_dict)
            return {
                "input": rewritten,
                "chat_history": input_dict["chat_history"]
            }

        rag_chain = RunnableLambda(rewrite_and_pass) | question_answer_chain

        return RunnableWithMessageHistory(
            rag_chain,
            self._get_history,
            input_messages_key="input",
            history_messages_key="chat_history"
        )



