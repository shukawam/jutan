import os
from dotenv import find_dotenv, load_dotenv
import streamlit as st
from agent import JutanAgent
from traceloop.sdk import Traceloop

_ = load_dotenv(find_dotenv())
api_endpoint = os.getenv("TRACELOOP_BASE_URL")

Traceloop.init(
    # デモ用なので、batch processorではなく即時でトレースデータを送る
    disable_batch=True,
    # アプリケーションの名前
    app_name="NVIDIA AI Summit Session Bot",
    # 独自属性の追加
    resource_attributes={"env": "demo", "version": "1.0.0"},
    api_endpoint="api_endpoint",
)

st.title(body="Jutan a.k.a Extended RAG(RUG)")
st.caption(
    body="""
    Jutan is as known as an extended RAG(RUG) application that included agent, many search options and observability.
"""
)

# サイドバー関連
with st.sidebar.container():
    with st.sidebar:
        st.sidebar.markdown("### LLM関連パラメータ")
        model_name = st.sidebar.selectbox(
            label="LLM Provider",
            options=["oci", "openai", "cohere"],
        )
        max_tokens = st.sidebar.slider(
            label="Max Tokens",
            min_value=128,
            max_value=2048,
            value=1024,
            step=128,
            help="LLMが出力する最大のトークン長",
        )
        temperature = st.sidebar.slider(
            label="Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="モデルの出力のランダム性",
        )
        st.sidebar.markdown("### 検索関連パラメータ")
        top_k = st.sidebar.slider(
            label="Top K",
            min_value=1,
            max_value=50,
            value=25,
            step=1,
            help="関連情報の取得数",
        )
        use_reranker = st.sidebar.radio(
            label="Reranker",
            options=[True, False],
            horizontal=True,
            help="Rerankerを使用するか（※使用には、CohereのAPI Keyが必要です）",
        )
        top_n = st.sidebar.slider(
            label="Top N",
            min_value=1,
            max_value=25,
            value=5,
            help="Rerankerで取得した情報を何件に絞り込むか",
        )

if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("どんなセッションが聞きたいですか？"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        messages = [
            {"role": message["role"], "content": message["content"]}
            for message in st.session_state.messages
        ]
        agent = JutanAgent(model_name=model_name)
        response = agent.run(
            question=prompt, top_k=top_k, use_reranker=use_reranker, top_n=top_n
        )
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
