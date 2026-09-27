# ui.py
import streamlit as st
import requests

st.set_page_config(
    page_title="MINAGRI Policy AI Assistant",
    layout="wide"
)

st.title("MINAGRI Policy Assistant")
st.caption("Ask questions regarding Rwandan agricultural strategies, livestock development, and policy frameworks.")

with st.sidebar:
    st.header("Ingested Policy Documents")
    st.markdown("- **PSTA 4** (Strategic Plan for Agriculture Transformation)")
    st.markdown("- **PSTA 5** (Fulltext Strategy)")
    st.markdown("- **Livestock Development Strategy Final**")
    st.markdown("- **Leveraging Private Sector Strategy**")
    st.divider()
    st.info("System connected to ChromaDB and Groq LLM API.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_prompt := st.chat_input("Ask a question about MINAGRI policy..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        def response_generator():
            try:
                res = requests.post(
                    "http://localhost:8000/api/v1/query",
                    json={"query": user_prompt},
                    stream=True
                )
                if res.status_code == 200:
                    for chunk in res.iter_content(chunk_size=None, decode_unicode=True):
                        if chunk:
                            yield chunk
                else:
                    yield f"Error: Server returned status code {res.status_code}"
            except Exception as e:
                yield f"Failed to connect to backend: {str(e)}"

        # st.write_stream consumes the generator and handles real-time UI rendering
        full_response = st.write_stream(response_generator)

    st.session_state.messages.append({"role": "assistant", "content": full_response})