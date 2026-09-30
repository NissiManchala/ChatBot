"""Optional Streamlit UI.  Run from the project root:  streamlit run streamlit_app.py"""
import streamlit as st

from src.graph import build_rag_graph

st.set_page_config(page_title="Agentic AI eBook Chatbot", page_icon="🤖", layout="wide")


@st.cache_resource
def get_graph():
    return build_rag_graph()


st.title("🤖 Agentic AI eBook Chatbot")
st.caption("Answers are grounded strictly in *Agentic AI for Executives* (Konverge AI).")

if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role", "content"}]
if "last" not in st.session_state:
    st.session_state.last = None  # last graph result, shown in the side panel

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

if query := st.chat_input("Ask something about the eBook..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)
    with st.chat_message("assistant"):
        with st.spinner("Searching the eBook..."):
            result = get_graph().invoke({"question": query, "context": [], "answer": "", "score": 0.0})
        st.write(result["answer"])
    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
    st.session_state.last = result

# Side panel: retrieved chunks + score for the latest answer
with st.sidebar:
    st.header("Retrieval details")
    last = st.session_state.last
    if not last:
        st.info("Ask a question to see the retrieved context here.")
    else:
        st.metric("Confidence score", f"{last['score']:.2f}")
        st.subheader(f"Retrieved chunks ({len(last['context'])})")
        for i, c in enumerate(last["context"], 1):
            with st.expander(f"{i}. Page {c['page']} · similarity {c['score']:.3f}"):
                st.write(c["text"])
