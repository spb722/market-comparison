import streamlit as st
import uuid
import pandas as pd
from typing import List, Dict, Any
import time
import json

# Import necessary modules from the chatbot code
# Note: Make sure your LangGraph chatbot code is in a file named 'telecom_bot.py'
from main import build_graph, GraphState

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

if "extraction_completed" not in st.session_state:
    st.session_state.extraction_completed = False

if "telecom_data" not in st.session_state:
    st.session_state.telecom_data = None

if "last_response" not in st.session_state:
    st.session_state.last_response = None

# Set up the Streamlit page
st.set_page_config(page_title="Telecom Product Chatbot", page_icon="📱")
st.title("Telecom Product Chatbot")

# Create sidebar with information
with st.sidebar:
    st.header("About")
    st.markdown("""
    This chatbot helps you find telecom products based on your needs.

    **Features:**
    - Search for telecom bundles
    - Get product recommendations
    - Select a product to purchase

    **Example queries:**
    - "Show me the latest Telecel Zimbabwe data bundles"
    - "I need a voice and data combo bundle"
    - "What are the cheapest daily data options?"
    """)

    # Reset conversation button
    if st.button("Reset Conversation"):
        st.session_state.messages = []
        st.session_state.conversation_id = str(uuid.uuid4())
        st.session_state.extraction_completed = False
        st.session_state.telecom_data = None
        st.session_state.last_response = None
        st.rerun()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Display telecom data table if available - this needs to be after messages but before input
if st.session_state.telecom_data is not None:
    st.subheader("Available Telecom Plans")
    st.dataframe(st.session_state.telecom_data, use_container_width=True)
    st.markdown("---")  # Add a separator

# Get user input
prompt = st.chat_input("Ask about telecom products...")

# Process user input
if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response with a spinner while processing
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")

        try:
            # Prepare input for the graph
            input_data = {
                "query": prompt,
                "extraction_completed": st.session_state.extraction_completed,
                "conversation_id": st.session_state.conversation_id,
                "human_type": "customer"
            }

            # If this is a follow-up and we have formatted text from the previous response,
            # make sure to include it in the input data
            if st.session_state.extraction_completed and st.session_state.last_response:
                input_data["formatted_text"] = st.session_state.last_response.get("formatted_text", "")
                input_data["followup_type"] = "selection"  # Set followup_type for product selection

            # Show a progress indicator during processing
            with st.spinner("Processing..."):
                # Invoke the graph with the input data
                thread_config = {"configurable": {"thread_id": st.session_state.conversation_id}}
                result = st.session_state.graph.invoke(input_data, config=thread_config)

                # Save the result for potential future reference
                st.session_state.last_response = result

                # Handle telecom data if present
                if "formatted_data" in result:
                    telecom_plans = result.get("formatted_data")
                    # Create a list of dictionaries manually
                    telecom_data = []
                    for plan in telecom_plans:
                        telecom_data.append({
                            'Provider': plan.Provider,
                            'Plan_Details': plan.Plan_Details,
                            'Price': plan.Price,
                            'Observations': plan.Observations,
                            'Recommended_Action_for_Econet': plan.Recommended_Action_for_Econet,
                            'Best_Suited_Econet_Plan_Equivalent': plan.Best_Suited_Econet_Plan_Equivalent
                        })
                    # Create the DataFrame
                    st.session_state.telecom_data = pd.DataFrame(telecom_data)
                    # Log that we've updated the DataFrame
                    print(f"Updated telecom_data with {len(telecom_data)} records")

                # Extract the response message
                if "messages" in result and result["messages"]:
                    # Get the most recent message
                    if isinstance(result["messages"][0], str):
                        bot_response = result["messages"][-1]
                    else:
                        # If it's not a string, try to get the content property
                        bot_response = getattr(result["messages"][-1], "content", str(result["messages"][-1]))

                    # For debugging
                    print(f"Received message from graph: {bot_response[:100]}...")
                else:
                    bot_response = "I'm sorry, I couldn't process that request."

                # Update extraction_completed state if present in result
                if "extraction_completed" in result:
                    st.session_state.extraction_completed = result["extraction_completed"]

                # Log action if present (for debugging)
                if "action" in result:
                    st.session_state.action = result["action"]

                # Update the followup_type if present
                if "followup_type" in result:
                    st.session_state.followup_type = result["followup_type"]

                # Update the message placeholder with the response
                message_placeholder.markdown(bot_response)

                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": bot_response})

                # Only force a rerun if this is the first extraction (not a follow-up)
                if "formatted_data" in result and not st.session_state.extraction_completed:
                    st.rerun()

        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            message_placeholder.markdown(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            st.error(error_message)

# Add some styling
st.markdown("""
<style>
.stApp {
    max-width: 1000px;
    margin: 0 auto;
}
.stDataFrame {
    margin-top: 20px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)