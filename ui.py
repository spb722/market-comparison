import streamlit as st
import uuid
import pandas as pd
from typing import List, Dict, Any
import time
import json
import os
from PIL import Image

# Import necessary modules from the chatbot code
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

if "show_table" not in st.session_state:
    st.session_state.show_table = False

if "initialized" not in st.session_state:
    st.session_state.initialized = False

# Set up the Streamlit page
st.set_page_config(page_title="Market Analysis", page_icon="📊")

# Display logo from file path
try:
    # Define the path to your logo image relative to this file
    # Replace 'logo.png' with your actual logo filename
    logo_path = "logo.png"  # You'll need to place your logo.png file in the same directory as this script

    # Check if the file exists
    if os.path.exists(logo_path):
        # Display the logo image
        logo = Image.open(logo_path)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(logo, use_container_width=True)
    else:
        st.title("Market Analysis")
        st.warning(f"Logo file not found at {logo_path}")
except Exception as e:
    st.title("Market Analysis")
    st.error(f"Error loading logo: {str(e)}")

# Add subtitle
st.markdown(
    "<h3 style='text-align: center; color: #666; margin-top: -10px;'>AARYA - AUTO AI RESPONDER AT YOUR ASSISTANCE</h3>",
    unsafe_allow_html=True)
st.markdown("---")

# Create sidebar with information
with st.sidebar:
    st.header("About")
    st.markdown("""
    This AI assistant helps you analyze telecom market data and products.

    **Features:**
    - Competitor analysis
    - Product comparisons
    - Strategic recommendations

    **Example queries:**
    - "Show me the latest Telecel Zimbabwe data bundles"
    - "Compare NetOne and Telecel voice packages"
    - "What should Econet do to compete with NetOne's pricing?"
    """)

    # Reset conversation button
    if st.button("Reset Conversation"):
        st.session_state.messages = []
        st.session_state.conversation_id = str(uuid.uuid4())
        st.session_state.extraction_completed = False
        st.session_state.telecom_data = None
        st.session_state.last_response = None
        st.session_state.show_table = False
        st.session_state.initialized = False
        st.rerun()

# Initialize with welcome message
if not st.session_state.initialized:
    welcome_message = """
    👋 Welcome to the Telecom Market Analysis assistant!

    What would you like me to analyze today?
    """
    st.session_state.messages.append({"role": "assistant", "content": welcome_message})
    st.session_state.initialized = True

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Display telecom data table if available
if st.session_state.telecom_data is not None and st.session_state.show_table:
    st.subheader("Competitive Analysis")
    st.dataframe(st.session_state.telecom_data, use_container_width=True)
    st.markdown("---")  # Add a separator

# Get user input
prompt = st.chat_input("Ask about telecom market and products...")

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
        message_placeholder.markdown("Analyzing market data...")

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
            with st.spinner("Analyzing telecom market data..."):
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
                    st.session_state.show_table = True
                    print(f"Updated telecom_data with {len(telecom_data)} records")

                # Extract the response message
                if "messages" in result and result["messages"]:
                    # Get the most recent message
                    if isinstance(result["messages"][-1], str):
                        bot_response = result["messages"][-1]
                    else:
                        # If it's not a string, try to get the content property
                        bot_response = getattr(result["messages"][-1], "content", str(result["messages"][-1]))

                    print(f"Received message from graph: {bot_response[:100]}...")
                else:
                    bot_response = "I'm sorry, I couldn't process that market analysis request."

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

                # Force a rerun to show the table immediately after first extraction
                if "formatted_data" in result and st.session_state.show_table:
                    st.rerun()

        except Exception as e:
            error_message = f"An error occurred during analysis: {str(e)}"
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
.stChatMessage {
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)
