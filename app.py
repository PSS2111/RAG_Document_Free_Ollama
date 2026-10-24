import io
import requests
import numpy as np
import streamlit as st
from pypdf import PdfReader

st.set_page_config(page_title="Free Local RAG", page_icon="📚", layout="wide")
st.title("📚 Free Local RAG Document Q&A")
st.write("Upload a PDF/TXT and ask questions using models running locally through Ollama.")
st.caption("No paid API key. Document text, embeddings, and generation stay on your machine.")

with st.sidebar:
    st.header("Ollama settings")
    url=st.text_input("Ollama URL","http://localhost:11434").strip().rstrip("/")
    chat_model=st.text_input("Chat model","llama3.2:3b").strip()
    embed_model=st.text_input("Embedding model","nomic-embed-text").strip()
    size=st.slider("Chunk size",500,1800,1000,100)
    overlap=st.slider("Chunk overlap",0,300,150,50)
    top_k=st.slider("Retrieved chunks",1,6,4)

def extract(f):
    if f.name.lower().endswith(".pdf"):
        reader=PdfReader(io.BytesIO(f.getvalue()))
        return "\n".join(f"[Page {i+1}]\n{p.extract_text() or ''}" for i,p in enumerate(reader.pages))
    return f.getvalue().decode("utf-8",errors="replace")

def chunks_of(text,n,overlap):
    text=" ".join(text.split())
    step=max(1,n-overlap)
    return [text[i:i+n].strip() for i in range(0,len(text),step) if text[i:i+n].strip()]

def embed(url,model,texts):
    out=[]
    for i in range(0,len(texts),32):
        r=requests.post(url+"/api/embed",json={"model":model,"input":texts[i:i+32]},timeout=300)
        r.raise_for_status()
        out.extend(r.json()["embeddings"])
    return np.asarray(out,dtype=np.float32)

def retrieve(url,model,q,chunks,vecs,k):
    r=requests.post(url+"/api/embed",json={"model":model,"input":[q]},timeout=120)
    r.raise_for_status()
    qv=np.asarray(r.json()["embeddings"][0],dtype=np.float32)
    denom=np.maximum(np.linalg.norm(vecs,axis=1)*np.linalg.norm(qv),1e-12)
    scores=vecs@qv/denom
    ids=np.argsort(scores)[::-1][:k]
    return [(int(i),float(scores[i]),chunks[int(i)]) for i in ids]

def answer(url,model,q,matches):
    context="\n\n".join(f"[Source chunk {i+1}]\n{t}" for i,s,t in matches)
    prompt=("Answer only from the document context. If it doesn't contain the answer, say so. "
            "Do not invent facts. End with the source chunk numbers used.\n\n"
            f"CONTEXT:\n{context}\n\nQUESTION:\n{q}")
    r=requests.post(url+"/api/chat",json={"model":model,"messages":[
        {"role":"system","content":"You are a careful document Q&A assistant. Be concise and grounded."},
        {"role":"user","content":prompt}], "stream":False,"options":{"temperature":0.1}},timeout=600)
    r.raise_for_status()
    return r.json()["message"]["content"]

try:
    r=requests.get(url+"/api/tags",timeout=4); r.raise_for_status()
    names=[m.get("name","") for m in r.json().get("models",[])]
    st.sidebar.success("Ollama is running")
    missing=[m for m in (chat_model,embed_model) if not any(n==m or n.startswith(m+":") for n in names)]
    if missing:
        st.sidebar.warning("Download missing models in a terminal:")
        st.sidebar.code("\n".join("ollama pull "+m for m in missing))
except Exception:
    st.sidebar.error("Ollama not reachable. Install/start it: https://ollama.com/download")

if "chunks" not in st.session_state:
    st.session_state.chunks=None
    st.session_state.vectors=None
    st.session_state.filename=None

f=st.file_uploader("Upload document",type=["pdf","txt"])
if st.button("Process document",type="primary"):
    if not f: st.error("Upload a PDF or TXT first.")
    elif overlap>=size: st.error("Overlap must be smaller than chunk size.")
    else:
        try:
            with st.spinner("Extracting text and embedding chunks locally..."):
                ch=chunks_of(extract(f),size,overlap)
                if not ch: st.error("No readable text found. Scanned PDFs need OCR first.")
                else:
                    v=embed(url,embed_model,ch)
                    st.session_state.chunks=ch
                    st.session_state.vectors=v
                    st.session_state.filename=f.name
                    st.success(f"Indexed {len(ch)} chunks from {f.name}.")
        except Exception as e:
            st.error("Processing failed. Check that Ollama is running and the embedding model is installed.")
            st.exception(e)

if st.session_state.chunks:
    st.subheader("Ask about: "+st.session_state.filename)
    q=st.text_input("Question",placeholder="What are the main points?")
    if st.button("Ask question"):
        if not q.strip(): st.warning("Enter a question.")
        else:
            try:
                with st.spinner("Retrieving passages and generating a local answer..."):
                    matches=retrieve(url,embed_model,q,st.session_state.chunks,st.session_state.vectors,top_k)
                    result=answer(url,chat_model,q,matches)
                st.markdown("### Answer"); st.write(result)
                with st.expander("Inspect retrieved source chunks"):
                    for i,s,t in matches:
                        st.markdown(f"**Chunk {i+1} · similarity {s:.3f}**"); st.write(t); st.divider()
            except Exception as e:
                st.error("Answer failed. Check Ollama and both model names.")
                st.exception(e)

st.divider()
st.write("**RAG:** extract text → chunk → local embeddings → cosine similarity retrieval → local LLM answer.")
