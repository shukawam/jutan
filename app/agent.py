import os, logging
from dotenv import find_dotenv, load_dotenv
from vector_store import VectorStore
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai.chat_models import ChatOpenAI
from langchain_cohere.chat_models import ChatCohere
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI

_ = load_dotenv(find_dotenv())
logger = logging.getLogger(__name__)


class JutanAgent:
    def __init__(self, model_name: str = "openai") -> None:
        self.model_name = model_name
        if "openai".__eq__(model_name):
            openai_api_key = os.getenv("OPENAI_API_KEY")
            self.llm = ChatOpenAI(api_key=openai_api_key, model="got-4o")
        elif "cohere".__eq__(model_name):
            cohere_api_key = os.getenv("COHERE_API_KEY")
            self.llm = ChatCohere(cohere_api_key=cohere_api_key, model="command-r-plus")
        elif "oci".__eq__(model_name):
            compartment_id = os.getenv("COMPARTMENT_ID")
            self.llm = ChatOCIGenAI(
                auth_type="INSTANCE_PRINCIPAL",
                model_id="cohere.command-r-plus-08-2024",
                compartment_id=compartment_id,
                service_endpoint="https://inference.generativeai.ap-osaka-1.oci.oraclecloud.com",
                model_kwargs={"max_tokens": 1024, "temperature": 0.7},
            )
        else:
            logger.error("Unsetted model name.")

    def run(
        self, question: str, top_k: int = 25, use_reranker: bool = True, top_n: int = 5
    ) -> str:
        vector_store_option = os.getenv("VECTOR_STORE")
        user = os.getenv("USER")
        password = os.getenv("PASSWORD")
        dsn = os.getenv("DSN")
        config_dir = os.getenv("CONFIG_DIR")
        wallet_location = os.getenv("WALLET_LOCATION")
        wallet_password = os.getenv("WALLET_PASSWORD")
        vector_store = VectorStore(
            model_name=self.model_name,
            vector_store_option=vector_store_option,
            user=user,
            password=password,
            dsn=dsn,
            config_dir=config_dir,
            wallet_location=wallet_location,
            wallet_password=wallet_password,
        )
        retriever = vector_store.get_retriever(top_k=top_k, use_reranker=use_reranker, top_n=top_n)
        keyword_gen_prompt = ChatPromptTemplate.from_template(
            template="""
            あなたは、AI分野に詳しい専門家です。
            与えられた質問に対して関連するキーワードを3つ連想してください。
            回答は、以下の形式に則って出力してください。
            
            ## 質問
            {question}
            
            ## 出力形式
            キーワード1, キーワード2, キーワード3
            """
        )
        compartment_id = os.getenv("COMPARTMENT_ID")
        keyword_gen_llm = ChatOCIGenAI(
            auth_type="INSTANCE_PRINCIPAL",
            model_id="meta.llama-3.1-70b-instruct",
            compartment_id=compartment_id,
            service_endpoint="https://inference.generativeai.ap-osaka-1.oci.oraclecloud.com",
            model_kwargs={"max_tokens": 1024, "temperature": 0.7},
        )
        prompt = ChatPromptTemplate.from_template(
            template="""
            あなたは、質問者に適切なセッションをガイドする案内人です。
            与えられたキーワードに類似するセッションを検索によって得られたセッションリストのみから抽出して案内してください。
            セッション案内は、検索によって得られたセッションリスト以外を用いてはいけませんし、嘘の情報を出力することもいけません。
            また、質問者にとってどの程度おすすめできるかのスコア(10段階評価)とその理由も算出してください。
            提案は、最大3セッションまでとしてください。
            出力は、以下に指定するフォーマット通り出力してください。
            
            ---
            
            キーワード:
            {keyword}
            
            コンテキスト:
            {context}
            
            出力フォーマット:
            #### セッションタイトル
            <セッションのタイトル>
            
            #### セッション概要
            <セッションの概要>
            
            #### おすすめ度
            <スコア>/10
            <理由>
            """
        )
        responder_chain = (
            {
                "keyword": (
                    {"question": RunnablePassthrough()}
                    | keyword_gen_prompt
                    | keyword_gen_llm
                    | StrOutputParser()),
                "context": retriever}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return responder_chain.invoke(input=question)
