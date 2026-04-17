import streamlit as st
import requests

st.set_page_config(page_title="Logistics AI", layout="wide")
st.title("🚛 Logistics Document Processor")

if "doc_id" not in st.session_state:
    st.session_state.doc_id = None

# 1. Upload Section
uploaded_file = st.file_uploader("Upload Rate Confirmation / Invoice", type=["pdf", "docx", "txt"])

if uploaded_file and st.button("Process Document"):
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        res = requests.post("http://localhost:8000/upload", files=files)
        if res.status_code == 200:
            st.session_state.doc_id = res.json()["doc_id"]
            st.success(f"Document Processed! ID: {st.session_state.doc_id}")
        else:
            st.error(f"Upload failed: {res.text}")
    except Exception as e:
        st.error(f"Could not connect to backend: {e}")

# 2. QA & Extraction Tabs
if st.session_state.doc_id:
    tab1, tab2 = st.tabs(["💬 Ask Questions", "📊 Structured Extraction"])
    
    with tab1:
        query = st.text_input("Ask something (e.g., 'What is the carrier rate?')")
        if query:
            with st.spinner("Searching document..."):
                try:
                    # Added timeout because LLMs take time to load on 3GB RAM
                    res = requests.post(
                        "http://localhost:8000/ask", 
                        json={"doc_id": st.session_state.doc_id, "question": query},
                        timeout=120 
                    )
                    if res.status_code == 200:
                        data = res.json()
                        st.metric("Confidence Score", data.get("confidence", 0))
                        st.write(f"**Answer:** {data.get('answer', 'No answer found')}")
                        with st.expander("View Supporting Source Text"):
                            st.write(data.get("sources", "No sources provided"))
                    else:
                        st.error(f"Error from server: {res.status_code}")
                except requests.exceptions.Timeout:
                    st.error("The request timed out. The model is likely still loading in RAM.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

    with tab2:
        if st.button("Run Extraction"):
            with st.spinner("Extracting structured data... This may take 30-60 seconds..."):
                try:
                    # Increased timeout for complex extraction
                    res = requests.post(
                        "http://localhost:8000/extract", 
                        json={"doc_id": st.session_state.doc_id, "question": ""},
                        timeout=180
                    )
                    
                    if res.status_code == 200:
                        # Safely try to parse JSON
                        try:
                            st.json(res.json())
                        except ValueError:
                            st.error("Backend returned an invalid response (not JSON).")
                    else:
                        st.error(f"Extraction failed. Server returned: {res.status_code}")
                        st.info("Check your VS Code backend terminal for RAM errors.")
                        
                except requests.exceptions.Timeout:
                    st.error("Extraction timed out. Your PC needs more time to process the AI model.")
                except Exception as e:
                    st.error(f"Connection error: {e}")
