import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma

from langchain_community.vectorstores import FAISS  

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import trim_messages

from pathlib import Path

api_key = os.getenv("OPENAI_API_KEY")

def find_file_path(file_name):
    script_dir = Path(__file__).resolve().parent

    for base_dir in [script_dir, *script_dir.parents]:
        direct_path = base_dir / file_name
        if direct_path.is_file():
            return str(direct_path.resolve())

        for root, _, files in os.walk(base_dir):
            if file_name in files:
                return str((Path(root) / file_name).resolve())

    return None



@st.cache_resource
def process_pdf():
    file_path = find_file_path("2024_KB_부동산_보고서_최종.pdf")
    loader = PyPDFLoader(file_path=file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(docs)

@st.cache_resource
def initialize_vectorstore():
    embeddings = OpenAIEmbeddings(api_key=api_key)

    db_name = Path(__file__).resolve().parent / "faiss.index"
    if os.path.exists(db_name):
        faiss = FAISS.load_local(
            db_name,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        chunks = process_pdf()
        faiss = FAISS.from_documents(chunks, embeddings)
        faiss.save_local(db_name)

    return faiss

# 체인 초기화
@st.cache_resource
def initialize_chain():
    vectorstore = initialize_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    chat_history = ChatMessageHistory()

    template = """당신은 KB 부동산 보고서 전문가입니다. 다음 정보를 바탕으로 사용자의 질문에 답변해주세요.

    컨텍스트: {context}
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        ("placeholder", "{chat_history}"),
        ("human", "{question}")
    ])

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    base_chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["question"])),
            chat_history=lambda x: trim_messages(
                x["chat_history"],
                max_tokens=4,
                token_counter=len,
                strategy="last",
            )
        )
        | prompt
        | model
        | StrOutputParser()
    )

    return RunnableWithMessageHistory(
        base_chain,
        lambda session_id : chat_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

# Streamlit UI
def main():
    st.set_page_config(page_title="KB 부동산 보고서 챗봇", page_icon="🏠")
    st.title("🏠 KB 부동산 보고서 AI 어드바이저")
    st.caption("2024 KB 부동산 보고서 기반 질의응답 시스템")

    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 채팅 기록 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 입력 처리
    if prompt := st.chat_input("부동산 관련 질문을 입력하세요"):
        # 사용자 메시지 표시
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 체인 초기화
        chain = initialize_chain()

        # AI 응답 생성
        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                response = chain.invoke(
                    {"question": prompt},
                    {"configurable": {"session_id": "streamlit_session"}}
                )
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__=="__main__":
    main()
