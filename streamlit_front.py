import streamlit as st
import requests
import time

st.title("MATLAB RAG")

question = st.text_input("Ask a question about MATLAB:")

if st.button("Submit"):
    if question:
        # Start timing when user submits question
        start_time = time.time()
        
        with st.spinner("Processing your question..."):
            try:
                # Make API call to FastAPI backend
                response = requests.post(
                    "http://localhost:8000/generate",
                    json={"question": question}
                )
                
                # Calculate response time
                end_time = time.time()
                response_time = end_time - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display the answer
                    st.subheader("Answer:")
                    st.write(result["answer"])
                    
                    # Display response time
                    st.info(f"Answered in {response_time:.2f} seconds")
                    
                    # Display contexts in expandable section
                    with st.expander("View Source Contexts"):
                        for i, context in enumerate(result["contexts"], 1):
                            st.write(f"**Context {i}:**")
                            st.write(context)
                            st.write("---")
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
                    st.info(f"Failed after {response_time:.2f} seconds")
                    
            except requests.exceptions.ConnectionError:
                end_time = time.time()
                response_time = end_time - start_time
                st.error("Could not connect to the RAG backend. Make sure the FastAPI server is running on http://localhost:8000")
                st.info(f"Connection failed after {response_time:.2f} seconds")
            except Exception as e:
                end_time = time.time()
                response_time = end_time - start_time
                st.error(f"An error occurred: {str(e)}")
                st.info(f"Error occurred after {response_time:.2f} seconds")
    else:
        st.warning("Please enter a question before submitting.")