import os
import sys
import json

# if __name__ == "__main__":
#     sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from utils.config_loader import load_config
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import CustomException
from langchain_huggingface import HuggingFaceEmbeddings


class ApiKeyManager:
    # Class attribute
    REQUIRED_KEYS = ["GROQ_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "HUGGINGFACEHUB_API_TOKEN"]

    def __init__(self):
        self.api_keys = {}

        # Fallback to individual env vars
        for key in self.REQUIRED_KEYS:
            if not self.api_keys.get(key):
                env_val = os.getenv(key)
                if env_val:
                    self.api_keys[key] = env_val
                    log.info(f"Loaded {key} from individual env var")

        # Final check
        missing = [k for k in self.REQUIRED_KEYS if not self.api_keys.get(k)]
        if missing:
            log.error("Missing required API keys", missing_keys=missing)
            raise CustomException("Missing API keys", sys)

        log.info("API keys loaded", keys={k: v[:6] + "..." for k, v in self.api_keys.items()})


    def get(self, key: str) -> str:
        val = self.api_keys.get(key)
        if not val:
            raise KeyError(f"API key for {key} is missing")
        return val



class ModelLoader:
    """
    Loads embedding models and LLMs based on config and environment.
    """

    def __init__(self):
        if os.getenv("ENV", "local").lower() != "production":
            load_dotenv()
            log.info("Running in LOCAL mode: .env loaded")
        else:
            log.info("Running in PRODUCTION mode")

        self.api_key_mgr = ApiKeyManager()
        self.config = load_config()
        log.info("YAML config loaded", config_keys=list(self.config.keys()))

    def load_embeddings(self):
        """
        Load and return embedding model from Google Generative AI.
        """
        try:
            embedding_block = self.config["embedding_model"]
            provider_key = os.getenv("EMBEDDING_PROVIDER", "huggingface")

            if provider_key not in embedding_block:
                log.error("Embedding provider not found in config", provider=provider_key)
                raise ValueError(f"Embedding provider '{provider_key}' not found in config")
            
            embedding_config = embedding_block[provider_key]
            provider = embedding_config.get("provider")
            model_name = embedding_config.get("model_name")
            
            if provider == "google":
                return GoogleGenerativeAIEmbeddings(model=model_name,
                                                    google_api_key=self.api_key_mgr.get("GOOGLE_API_KEY")) #type: ignore

            elif provider == "openai":
                return OpenAIEmbeddings(model=model_name,
                                        openai_api_key=self.api_key_mgr.get("OPENAI_API_KEY")) #type: ignore
            elif provider == "huggingface":
                return HuggingFaceEmbeddings(model_name=model_name) #type: ignore
            else:
                log.error("Unsupported embedding provider", provider=provider)
                raise ValueError(f"Unsupported embedding provider: {provider}")

        except Exception as e:
            log.error("Error loading embedding model", error=str(e))
            raise CustomException("Failed to load embedding model", sys)

    def load_llm(self):
        """
        Load and return the configured LLM model.
        """
        llm_block = self.config["llm"]
        provider_key = os.getenv("LLM_PROVIDER", "groq")

        if provider_key not in llm_block:
            log.error("LLM provider not found in config", provider=provider_key)
            raise ValueError(f"LLM provider '{provider_key}' not found in config")

        llm_config = llm_block[provider_key]
        provider = llm_config.get("provider")
        model_name = llm_config.get("model_name")
        temperature = llm_config.get("temperature", 0.2)
        max_tokens = llm_config.get("max_output_tokens", 2048)

        log.info("Loading LLM", provider=provider, model=model_name)

        if provider == "google":
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=self.api_key_mgr.get("GOOGLE_API_KEY"),
                temperature=temperature,
                max_output_tokens=max_tokens
            )

        elif provider == "groq":
            return ChatGroq(
                model=model_name,
                api_key=self.api_key_mgr.get("GROQ_API_KEY"), #type: ignore
                temperature=temperature,
            )

        elif provider == "openai":
            return ChatOpenAI(
                model=model_name,
                api_key=self.api_key_mgr.get("OPENAI_API_KEY"),
                temperature=temperature,
                max_tokens=max_tokens
            )

        else:
            log.error("Unsupported LLM provider", provider=provider)
            raise ValueError(f"Unsupported LLM provider: {provider}")


if __name__ == "__main__":
    loader = ModelLoader()

    # Test Embedding
    embeddings = loader.load_embeddings()
    print(f"Embedding Model Loaded: {embeddings}")
    result = embeddings.embed_query("Hello, how are you?")
    print(f"Embedding Result: {result}")

    # Test LLM
    llm = loader.load_llm()
    print(f"LLM Loaded: {llm}")
    result = llm.invoke("Hello, how are you?")
    print(f"LLM Result: {result.content}")