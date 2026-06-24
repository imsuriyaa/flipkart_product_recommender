from enum import Enum
from typing import Dict
import string


class PromptType(str, Enum):
    PRODUCT_BOT = "product_bot"
    REWRITE_QUESTION_WITH_HISTORY = "rewrite_question_with_history"
    DETERMINISTIC_BOT = "deterministic_bot"
    # REVIEW_BOT = "review_bot"
    # COMPARISON_BOT = "comparison_bot"


class PromptTemplate:
    def __init__(self, template: str, description: str = "", version: str = "v1"):
        self.template = template.strip()
        self.description = description
        self.version = version

    def format(self, **kwargs) -> str:
        # Validate placeholders before formatting
        missing = [
            f for f in self.required_placeholders() if f not in kwargs
        ]
        if missing:
            raise ValueError(f"Missing placeholders: {missing}")
        return self.template.format(**kwargs)

    def required_placeholders(self):
        return [field_name for _, field_name, _, _ in string.Formatter().parse(self.template) if field_name]


# Central Registry
PROMPT_REGISTRY: Dict[PromptType, PromptTemplate] = {
    PromptType.PRODUCT_BOT: PromptTemplate(
        """
        You are an expert EcommerceBot specialized in product recommendations and handling customer queries.
        Analyze the provided product titles, ratings, and reviews to provide accurate, helpful responses.
        Stay relevant to the context, and keep your answers concise and informative.

        CONTEXT:
        {context}

        QUESTION: {question}

        YOUR ANSWER:
        """,
        description="Handles ecommerce QnA & product recommendation flows"
    )

    PromptType.REWRITE_QUESTION_WITH_HISTORY: PromptTemplate(
        """
        Given the conversation history and the latest user question,
        rewrite the latest question so it is completely self-contained.

        Do NOT answer the question.

        Latest question:
        {question}

        Standalone question:
        """
    )

    PromptType.DETERMINISTIC_BOT: PromptTemplate(
        """
        You have been provided with a set of product titles, ratings, and reviews. Your task is to determine whether the question is relevant to the provided data. Return ONLY YES or NO"

        Question: {question}

        if docs provided is not empty utilize it to determine the relevance of the question. If docs is empty go ahead with prompt in hand
        Product Data: {docs}
        """
    )


}