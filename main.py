
from typing import List, Dict, Any, Annotated, Optional
from typing_extensions import TypedDict
import operator
import json
import logging
import time
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
load_dotenv()


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from pydantic import BaseModel, Field, ValidationError

# APIs
from tavily import TavilyClient
from firecrawl import FirecrawlApp
import openai

# LangGraph
from langgraph.graph import StateGraph, START, END,MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.constants import Send
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import os

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
# Get model provider from environment variable, default to "groq" if not set
model_provider = os.environ.get("MODEL_PROVIDER", "groq").lower()

if model_provider == "groq":
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )
else:
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=os.environ.get("OPENAI_API_KEY")
    )
from langchain_ollama import ChatOllama
#
# llm = ChatOllama(
#     model="llama3.3:70b",
#     temperature=0,
#     # other params...
# )
def call_webhook(text: str, conversation_id: str) -> dict:
    """
    Call the Botpress webhook with the specified text and conversation ID.

    Args:
        text: The text to send in the payload
        conversation_id: The conversation ID to include in the request

    Returns:
        The response from the webhook as a dictionary or status information
    """
    import requests
    import logging
    print(f"calling webhook for {text}")
    webhook_url = "https://webhook.botpress.cloud/5c3a868e-d0f9-4505-9220-56b2fc854d1e"

    payload = {
        "payload": {
            "text": text
        },
        "conversationId": conversation_id
    }

    try:
        response = requests.post(webhook_url, json=payload)
        response.raise_for_status()  # Raise exception for 4XX/5XX responses

        # Try to parse JSON, but handle case where response isn't JSON
        try:
            return response.json()
        except ValueError:
            # Return status code and text if not JSON
            return {
                "status_code": response.status_code,
                "text": response.text,
                "message": "Response was not valid JSON"
            }

    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling webhook: {str(e)}")
        return {"error": str(e)}

class TelecomPlan(BaseModel):
    product_name: str = Field(description="The name of the product")
    product_description: Optional[str] = Field(None, description="Description of the product")
    product_offer_price: str = Field(description="The price of the product")
    voice_allowance: Optional[str] = Field(None, description="Voice allowance")
    sms_allowance: Optional[str] = Field(None, description="SMS allowance")


# State definitions
class SearchResult(TypedDict):
    url: str
    title: str
    content_snippet: str


class ExtractedContent(TypedDict):
    url: str
    content: str


class ProcessedData(TypedDict):
    url: str
    content: str
    extracted_data: Dict[str, Any]
    validation_result: Dict[str, Any]
    quality_score: float


# Graph state
class GraphState(MessagesState):
    query: str
    search_results: str
    extracted_contents: Annotated[List[ExtractedContent], operator.add]
    processed_data: Annotated[List[ProcessedData], operator.add]
    aggregated_results: Annotated[List[Dict[str, Any]],operator.add]
    formatted_text: Annotated[str,operator.add]
    extraction_completed: Annotated[bool ,operator.add]
    conversation_id: Annotated[str,operator.add]
    human_type : Annotated[str,operator.add]
    action: Annotated[str, operator.add]
    followup_type: str
    selected_product: str
    confirmation_message: str
    formatted_data: List[Any]

# 2. Create router function - split into node and condition functions
def router_node(state: GraphState) -> GraphState:
    """
    Router node function - must return a dictionary
    This is used as a node in the graph.
    """
    logging.info(f"Routing based on extraction_completed: {state.get('extraction_completed', False)}")
    # For nodes, we don't modify state, we just pass it through
    return {}

def router_condition(state: GraphState) -> str:
    """
    Router condition function - returns a string
    This is used for conditional edges.
    """
    if state.get("extraction_completed", False):
        return "follow_up"
    else:
        return "search"
# 2. Create router function
def route_to_search_or_followup(state: GraphState) -> str:
    """
    Route to search or follow-up based on extraction_completed state.
    """
    logging.info(f"Routing based on extraction_completed: {state.get('extraction_completed')}")

    if state.get("extraction_completed", False):
        return "follow_up"
    else:
        return "search"
class ProductSelection(BaseModel):
    selected_product: str = Field(description="The name of the product the user selected")
    confidence: float = Field(description="Confidence in the selection (0.0-1.0)")
    product_details: Dict[str, str] = Field(description="Extracted product details")


# 3. Create follow-up node
def follow_up(state: GraphState) -> GraphState:
    """
    Handle follow-up processing based on followup_type.
    """
    logging.info("Processing follow-up with existing extracted data")

    # Get the existing data
    query = state.get("query", "")
    followup_type = state.get("followup_type", "")
    formatted_text = state.get("formatted_text", "")

    # Preserve any existing state that needs to be maintained
    result = {
        "extraction_completed": state.get("extraction_completed", True),
        "formatted_data": state.get("formatted_data", [])  # Preserve formatted_data
    }

    logging.info(f"Follow-up processing of type {followup_type} for query: {query}")

    # Handle based on followup_type
    if followup_type == "selection" or "select" in query.lower() or "choose" in query.lower():
        # Process product selection
        user_feedback = query  # The user's selection feedback is in the query

        # Use structured output to extract the selection
        selection_prompt = f"""
        Based on the user's message, determine which Econet plan they want to select.

        Recommendations provided earlier:
        {formatted_text}

        User message:
        "{user_feedback}"

        Extract the name of the selected Econet plan. It should be one of the plans mentioned 
        in the recommendations. Look for phrases like "I want", "I'd like", "select", etc.

        Return the exact plan name as mentioned in the recommendations.
        """

        selection_response = llm.invoke(selection_prompt)
        selected_plan = selection_response.content.strip()

        logging.info(f"User selected plan: {selected_plan}")

        # Extract plan details from the formatted text
        extraction_prompt = f"""
        Extract all details for the plan named "{selected_plan}" from this text:

        {formatted_text}

        Return a detailed JSON structure with these fields (if available in the text):
        - plan_name: The exact name of the plan
        - description: Brief description of the plan
        - price: The price including currency symbol
        - validity: Validity period
        - data_allowance: Data allocation
        - voice_allowance: Voice minutes allocation
        - sms_allowance: SMS allocation
        - special_features: Any special features or perks

        If any information is not available, use "Not specified" as the value.

        Format your response as valid JSON only, with no additional text.
        """

        plan_details_response = llm.invoke(extraction_prompt)

        # Try to extract JSON from the response
        plan_details_text = plan_details_response.content.strip()
        if plan_details_text.startswith("```json"):
            plan_details_text = plan_details_text.replace("```json", "").replace("```", "").strip()
        elif plan_details_text.startswith("```"):
            plan_details_text = plan_details_text.replace("```", "").strip()

        try:
            plan_details = json.loads(plan_details_text)
            logging.info(f"Successfully extracted plan details: {plan_details}")
        except Exception as e:
            logging.error(f"Error parsing plan details: {str(e)}")
            # Create a fallback plan_details
            plan_details = {
                "plan_name": selected_plan,
                "description": "Not specified",
                "price": "Not specified",
                "validity": "Not specified",
                "data_allowance": "Not specified",
                "voice_allowance": "Not specified",
                "sms_allowance": "Not specified",
                "special_features": "Not specified"
            }

        # Generate confirmation message
        confirmation_prompt = f"""
        Create an enthusiastic confirmation message for a user who has selected the {selected_plan} plan.
        Here are the plan details:

        {json.dumps(plan_details, indent=2)}

        The message should:
        1. Start with an enthusiastic phrase like "Excellent choice!" or "Great selection!"
        2. Mention the name of the plan prominently
        3. Highlight the key advantages of this plan compared to others (using the plan details)
        4. Include specific benefits like data amount, voice minutes, and any special features
        5. End with the plan shall be created shortly 
        6. Be minimal 
        7. Format the message nicely with some emphasis on key details

        Use an excited, confident tone throughout as if this is definitely the best choice the user could make.
        DO NOT ask if this is the "correct" product or ask for a yes/no confirmation - instead ask about activation.
        """

        confirmation_response = llm.invoke(confirmation_prompt)
        confirmation_message = confirmation_response.content.strip()

        # Return just the confirmation message in the messages field
        result["followup_type"] = "confirmation"
        result["selected_product"] = plan_details
        result["messages"] = [confirmation_message]

        # Important: Make sure the original formatted_text is preserved for next interactions
        result["formatted_text"] = formatted_text

        return result

    elif followup_type == "confirmation" or "confirm" in query.lower() or "yes" in query.lower() or "proceed" in query.lower():
        # Process the confirmation response
        selected_product = state.get("selected_product", {})
        plan_name = selected_product.get("plan_name", "the selected plan")

        # Generate purchase completion message
        completion_prompt = f"""
        Create an exciting purchase completion message for a user who has activated 
        the {plan_name} plan.

        Plan details:
        {json.dumps(selected_product, indent=2)}

        The message should:
        1. Start with an enthusiastic congratulatory statement
        2. Announce that the {plan_name} plan has been successfully activated
        3. Highlight the main benefits they'll now enjoy (data, voice, etc.) with excitement
        4. Mention that they can start enjoying these benefits immediately
        5. Add a personalized touch about how this plan will enhance their communication experience
        6. Close with a friendly offer of further assistance or support
        7. Use emoji and formatting to make the message visually engaging

        Make it feel like a celebration of their excellent choice, not just a transaction confirmation.
        """

        completion_response = llm.invoke(completion_prompt)
        completion_message = completion_response.content.strip()

        # Return just the completion message in the messages field
        result["messages"] = [completion_message]
        result["action"] = "purchase_completed"
        result["formatted_text"] = formatted_text  # Preserve the original formatted text

        return result

    else:
        # Default case for general follow-up questions
        general_prompt = f"""
        The user has asked a follow-up question after being shown telecom plan recommendations.

        Previous recommendations:
        {formatted_text}

        User question: "{query}"

        Provide a helpful response that:
        1. Directly addresses their specific question
        2. References information from the recommended plans when relevant
        3. Is friendly and conversational
        4. Is concise (3-5 sentences)
        5. Offers to help them select a plan if they haven't already
        """

        general_response = llm.invoke(general_prompt)
        response_message = general_response.content.strip()

        # Return just the response message in the messages field
        result["messages"] = [response_message]
        result["formatted_text"] = formatted_text  # Preserve the original formatted text

        return result

# Function to search with Tavily
def search_with_tavily(state: GraphState) -> GraphState:
    """Search for relevant telecom information using OpenAI's search capability."""
    
    query = state["query"]
    print(f"searchin internet for {query} with openai web search")
    
    from openai import OpenAI
    client = OpenAI()
    
    completion = client.chat.completions.create(
        model="gpt-4o-search-preview",
        web_search_options={
            "search_context_size": "medium",
        },
        messages=[
            {
                "role": "assistant", 
                "content": "You are an expert telecom consultant working for Econet Zimbabwe. Your task is to analyze telecom product offerings from competitors like Telecel,"
                           " NetOne, and other telecom providers which the user shall provide found via internet search."
                           " Your goal is to find the best-priced offerings. "
        "You must present your findings clearly in a comparison table with columns for Provider, Plan Details, Price, Observations, "
        "Recommended Action for Econet, and Best Suited Econet Plan Equivalent. Provide clear explanations and actionable insights."
            },
            {
                "role": "user",
                "content": query + " Your goal is to find the best-priced offerings. "
        "You must present your findings clearly in a comparison table with columns for Provider, Plan Details, Price, Observations, "
        "Recommended Action for Econet, and Best Suited Econet Plan Equivalent. Provide clear explanations and actionable insights."
                "do multiple web search to get all the products "
            }
        ],
    )
    
    result = completion.choices[0].message.content
    return {"search_results": result}




class TelecomPlan(BaseModel):
    Provider: str = Field(description="Name of the telecom provider")
    Plan_Details: str = Field(description="Details of the telecom plan")
    Price: str = Field(description="Price of the plan")
    Observations: str = Field(description="Key observations about the plan")
    Recommended_Action_for_Econet: str = Field(description="Recommended action for Econet based on this plan")
    Best_Suited_Econet_Plan_Equivalent: str = Field(description="Best suited equivalent Econet plan")

class TelecomPlans(BaseModel):
    plans: List[TelecomPlan]
# Function to extract content from URL
def extract_content(state: Dict[str, str]) -> Dict[str, Any]:
    search_results = state["search_results"]
    prompt = (
        "You are an expert telecom data analyst. Structure the raw telecom data into a structured list of telecom plans as JSON. "
        "Each entry must contain Provider, Plan_Details, Price, Observations, Recommended_Action_for_Econet, and Best_Suited_Econet_Plan_Equivalent."
        "these are the search results  "
    )
    messages = [{"role": "system", "content": prompt}] + [search_results]
    structured_response = llm.with_structured_output(TelecomPlans).invoke(messages)

    # Check for missing or empty Best_Suited_Econet_Plan_Equivalent fields
    plans = structured_response.plans
    updated_plans = []

    for plan in plans:
        # Check if Best_Suited_Econet_Plan_Equivalent is missing or empty
        if not hasattr(plan,
                       'Best_Suited_Econet_Plan_Equivalent') or not plan.Best_Suited_Econet_Plan_Equivalent or plan.Best_Suited_Econet_Plan_Equivalent == "N/A":
            # Create prompt for the LLM to generate Econet plan equivalent
            econet_plan_prompt = f"""
            As a telecom expert, suggest the best Econet plan equivalent for this competitor plan:

            Provider: {plan.Provider}
            Plan Details: {plan.Plan_Details}
            Price: {plan.Price}
            Observations: {plan.Observations}
            Recommended Action for Econet: {plan.Recommended_Action_for_Econet}

            Based on the above information, provide a detailed description of an equivalent or better Econet plan.
            Include specific details like data allowance, voice minutes, SMS, validity period, and any special features.
            If Econet doesn't have a directly comparable plan, suggest the closest alternative or what Econet should offer.

            Respond ONLY with the plan details, formatted as a concise yet descriptive paragraph.
            """

            # Call LLM to generate Econet plan equivalent
            econet_plan_response = llm.invoke(econet_plan_prompt)

            # Update the plan with the generated equivalent
            plan_dict = {
                'Provider': plan.Provider,
                'Plan_Details': plan.Plan_Details,
                'Price': plan.Price,
                'Observations': plan.Observations,
                'Recommended_Action_for_Econet': plan.Recommended_Action_for_Econet,
                'Best_Suited_Econet_Plan_Equivalent': econet_plan_response.content.strip()
            }
            updated_plans.append(TelecomPlan(**plan_dict))
        else:
            # Keep the original plan if Best_Suited_Econet_Plan_Equivalent is present
            updated_plans.append(plan)

    return {"formatted_data": updated_plans, "messages": state.get("messages", [])}


# def process_data(state: GraphState) -> GraphState:
#     """Extract structured telecom bundle data from content."""
#     extracted_contents = state["extracted_contents"]
#     processed_data = []
#
#     logging.info(f"Processing data from {len(extracted_contents)} sources")
#
#     client = openai.OpenAI(
#         base_url="https://api.groq.com/openai/v1",
#         api_key="YOUR_API_KEY"
#     )
#
#     extraction_model = llm.with_structured_output(TelecomPlan)
#
#     for content_item in extracted_contents:
#         url = content_item["url"]
#         content = content_item["content"]
#
#         logging.info(f"Extracting structured data from {url}")
#
#         prompt = f"""
#         Extract telecom bundle details from the provided content:
#
#         Identify these fields explicitly if present:
#         - product_name
#         - product_description
#         - product_offer_price
#         - voice_allowance
#         - sms_allowance
#         - data_allowance
#         - validity
#
#         Content:
#         {content}  # Limit content length
#         """
#
#         try:
#             extracted_data = extraction_model.invoke(prompt)
#             time.sleep(0.3)
#             fields_present = sum(
#                 1 for field, value in extracted_data.model_dump().items()
#                 if value is not None and value != ""
#             )
#             total_fields = len(extracted_data.model_dump())
#             quality_score = fields_present / total_fields * 10
#
#             logging.info(f"Extracted data from {url}, quality score: {quality_score:.1f}/10")
#
#             processed_data.append({
#                 "url": url,
#                 "content": content,
#                 "extracted_data": extracted_data.model_dump(),
#                 "validation_result": {"valid": True, "message": "Data valid"},
#                 "quality_score": quality_score
#             })
#
#         except ValidationError as e:
#             logging.warning(f"Validation error for {url}: {str(e)}")
#
#             processed_data.append({
#                 "url": url,
#                 "content": content,
#                 "extracted_data": {},
#                 "validation_result": {"valid": False, "message": str(e)},
#                 "quality_score": 0
#             })
#
#     logging.info(f"Processed {len(processed_data)} total items")
#
#     return {"processed_data": processed_data}

def format_recommendations(state: GraphState) -> GraphState:
    """
    Format extracted telecom plans into chatbot-friendly recommendations with
    observations and rationale based on the Best_Suited_Econet_Plan_Equivalent field.
    """
    formatted_data = state.get("formatted_data", [])
    query = state.get("query", "")

    logging.info("Formatting recommendations from formatted data")

    # No products found case
    if not formatted_data:
        state["messages"] = [
            "I couldn't find any telecom plans matching your criteria. Could you please try a different query?"]
        state["extraction_completed"] = True
        return state

    # Create a prompt for the LLM to generate formatted recommendations
    prompt = f"""
    You are a telecom product recommendation expert for Econet. Format the extracted telecom plans into recommendations 
    that will be shown directly to users in a chatbot interface.

    The following are competitor plans with their Econet equivalents:
    {json.dumps([{
        "Provider": plan.Provider,
        "Plan_Details": plan.Plan_Details,
        "Price": plan.Price,
        "Observations": plan.Observations,
        "Recommended_Action_for_Econet": plan.Recommended_Action_for_Econet,
        "Best_Suited_Econet_Plan_Equivalent": plan.Best_Suited_Econet_Plan_Equivalent
    } for plan in formatted_data], indent=2)}

    User query: "{query}"

    Select the top 3 most relevant Econet plans from the Best_Suited_Econet_Plan_Equivalent field that best match the user's request.
    If there are fewer than 3 plans, include all of them.

    FORMAT YOUR RESPONSE EXACTLY LIKE THIS EXAMPLE:

    **Here are the best curated Econet recommendations for you:**

    ### 1. Econet Smart Data Bundle
    *High-speed data with extended validity*
    * **$5.00** *(valid 30 days)*
    * 🌐 Data: 2 GB
    * 📲 WhatsApp & Facebook: Unlimited

    ### 2. Econet Voice & Data Combo
    *Perfect balance of calls and internet*
    * **$10.00** *(valid 30 days)*
    * 📞 Minutes: 100 on-net, 50 off-net
    * 🌐 Data: 1.5 GB
    * 💬 SMS: 50

    ### 3. Econet Premium Bundle
    *Our most comprehensive package*
    * **$20.00** *(valid 30 days)*
    * 📞 Minutes: Unlimited on-net, 150 off-net
    * 🌐 Data: 5 GB
    * 💬 SMS: 100
    * 📲 Social Media: Unlimited

    **Observations:**
    Econet offers superior network coverage compared to competitors, with 98% population coverage nationwide. The recommended plans provide better value with longer validity periods and more flexible usage terms than similar competitor offerings.

    **Why These Plans:**
    These plans were selected because they offer the best match to your needs while providing better overall value compared to competitor options. Econet's Premium Bundle gives you significantly more data and minutes at a similar price point to competitor alternatives, while our Voice & Data Combo provides the most balanced allocation for typical users.

    YOUR RECOMMENDATIONS MUST FOLLOW THESE RULES:
    1. Include only actual Econet plans from the Best_Suited_Econet_Plan_Equivalent field
    2. If there are fewer than 3 plans, only include the actual plans you have data for
    3. Include a clear, bold product name with "###" heading
    4. Include an italicized short description
    5. Format prices clearly with **bold** text
    6. Use emojis for features:
       - 📞 for voice/calls
       - 💬 for SMS
       - 🌐 for data
       - 📲 for social media/WhatsApp
       - ⏱️ for validity period
    7. Include "Observations" section highlighting Econet's advantages over competitors
    8. Include "Why These Plans" section explaining why these Econet plans are better choices

    IMPORTANT GUIDELINES:
    - Focus on the strengths of Econet plans compared to competitors
    - Highlight any advantages in terms of price, data allocation, validity, or network quality
    - Make every recommendation sound enthusiastic and confident
    - Use the information from both the competitor plans and their Econet equivalents to make informed comparisons
    - The recommendations should feel like they were written by an Econet expert who genuinely believes these are excellent choices

    Your entire response must be in formatted markdown ready to display directly to the user.
    """

    try:
        # Use the LLM to generate the formatted text recommendations
        response = llm.invoke(prompt)
        recommendations = response.content.strip()

        logging.info("Successfully generated formatted recommendations")

    except Exception as e:
        logging.error(f"Error generating recommendations: {str(e)}")

        # Create fallback formatted text
        recommendations = "**Here are the best curated Econet recommendations for you:**\n\n"

        for i, plan in enumerate(formatted_data[:3], 1):
            econet_plan = plan.Best_Suited_Econet_Plan_Equivalent

            recommendations += f"### {i}. Econet Equivalent Plan\n"
            recommendations += f"*{econet_plan}*\n\n"

        recommendations += "**Observations:**\n"
        recommendations += "Econet offers superior network quality and reliability compared to competitors.\n\n"

        recommendations += "**Why These Plans:**\n"
        recommendations += "These Econet plans provide better overall value and coverage compared to competitor options."

    # Update state with the recommendations and mark extraction as completed
    state["messages"] = [recommendations]
    state["formatted_text"] = recommendations
    state["followup_type"] = "selection"
    state["extraction_completed"] = True

    return state
# def format_recommendations(state: GraphState) -> GraphState:
#     """
#     Format extracted products into chatbot-friendly recommendations with
#     observations and rationale as a formatted text string.
#     """
#     aggregated_results = state.get("aggregated_results", [])
#     query = state.get("query", "")
#
#     logging.info("Formatting recommendations from aggregated results")
#
#     # No products found case
#     if not aggregated_results:
#         return {
#             "formatted_text": "**No products found matching your criteria.**\n\nUnable to make recommendations due to insufficient data."
#         }
#
#     # Determine what kind of products the user is looking for based on the query
#     product_type = "bundle"  # Default to bundle products
#     if "data" in query.lower():
#         product_type = "data"
#     elif "voice" in query.lower() or "call" in query.lower():
#         product_type = "voice"
#
#     # Use the LLM to generate formatted recommendations
#     prompt = f"""
#     You are a telecom product recommendation expert. Format the extracted telecom products into a recommendation that will be shown directly to users in a chatbot interface.
#
#     I have extracted the following telecom products:
#     {json.dumps(aggregated_results, indent=2)}
#
#     User query: "{query}"
#
#     Select the top 3 most relevant {product_type} products that best match the user's request.
#     If there are fewer than 3 products, include all of them.
#
#     FORMAT YOUR RESPONSE EXACTLY LIKE THIS EXAMPLE:
#
#     **These are the recommended products:**
#
#     ### 1. MegaBoost Bundles
#     *Powerful combo bundles with Voice, SMS, Social Media & Data.*
#     * **$1.00 MegaBoost** *(valid 48 hrs)*
#     * 📞 On-net: 10 mins
#     * 💬 SMS: 30
#     * 📲 WhatsApp: 30 MB
#     * 👍 Facebook: 20 MB
#     * 🌐 Data: 15 MB
#
#     ### 2. Daily Data Bundle
#     *High-speed data for all your daily needs*
#     * **$0.50** *(valid 24 hrs)*
#     * 🌐 Data: 250 MB
#
#     ### 3. Cross-net Voice Bundles
#     *Call any network in Zimbabwe.*
#     * **ZIG 4** *(valid 24 hrs)*: 8 mins for ZIG 4
#     * **ZIG 9** *(valid 48 hrs)*: 18 mins for ZIG 9
#
#     **Observations:**
#     Telecel offers competitive pricing on combo bundles compared to other operators. Their daily data packages provide good value for users needing short-term connectivity. The voice bundles allow cross-network calling which is an advantage over network-restricted offers.
#
#     **Recommendation Rationale:**
#     These products were selected because they offer the best value for money in their respective categories. The MegaBoost Bundles provide an all-in-one solution for users who need multiple services. The Daily Data Bundle gives flexibility for temporary usage, and the Cross-net Voice Bundles solve the problem of calling across different networks.
#
#     YOUR RECOMMENDATIONS MUST FOLLOW THESE RULES:
#     1. Include only actual products - NEVER include placeholders like "No Product Available"
#     2. If there are fewer than 3 products, only include the actual products you have data for
#     3. Include a clear, bold product name with "###" heading
#     4. Include an italicized short description
#     5. Format prices clearly with **bold** text
#     6. Use emojis for features:
#        - 📞 for voice/calls
#        - 💬 for SMS
#        - 🌐 for data
#        - 📲 for social media/WhatsApp
#        - ⏱️ for validity period
#     7. Include "Observations" section with only positive market insights
#     8. Include "Recommendation Rationale" section explaining why these products are good choices
#
#     IMPORTANT GUIDELINES:
#     - Focus ONLY on the positive aspects of the products
#     - Do NOT mention limitations in the data or lack of product variety
#     - Do NOT apologize for incomplete information
#     - Do NOT include statements like "more products needed" or "limited options"
#     - If only one product is available, focus entirely on its benefits without mentioning the lack of alternatives
#     - Make every product sound like an excellent choice, no matter how limited the information
#     - Use confident, positive language throughout
#
#     The output should be formatted markdown text that can be displayed directly in a chat interface.
#     The response should feel like it was written by an enthusiastic telecom expert who genuinely believes these are excellent products.
#     Use the extracted data to inform your recommendations but feel free to format and present it in a more user-friendly way.
#
#     VERY IMPORTANT: Your entire response must be in plain text format ready to display. Do NOT include any JSON, code blocks, or other formatting that isn't part of the final user-facing text.
#     """
#
#     try:
#         # Use your model to generate the formatted text recommendations directly
#         response = llm.invoke(prompt)
#         formatted_text = response.content.strip()
#
#         logging.info("Successfully generated formatted text recommendations")
#
#     except Exception as e:
#         logging.error(f"Error generating recommendations: {str(e)}")
#
#         # Create fallback formatted text
#         formatted_text = "**These are the recommended products:**\n\n"
#
#         for i, product in enumerate(aggregated_results[:3], 1):
#             name = product.get("product_name", "Unnamed Product")
#             description = product.get("product_description", "No description available")
#             price = product.get("product_offer_price", "Price not available")
#
#             formatted_text += f"### {i}. {name}\n"
#             formatted_text += f"*{description}*\n"
#             formatted_text += f"* **{price}**\n"
#
#             if product.get("voice_allowance"):
#                 formatted_text += f"* 📞 Voice: {product.get('voice_allowance')}\n"
#
#             if product.get("sms_allowance"):
#                 formatted_text += f"* 💬 SMS: {product.get('sms_allowance')}\n"
#
#             if product.get("data_allowance"):
#                 formatted_text += f"* 🌐 Data: {product.get('data_allowance')}\n"
#
#             formatted_text += "\n"
#
#         formatted_text += "**Observations:**\n"
#         formatted_text += "These are excellent products that offer great value for Telecel customers.\n\n"
#
#         formatted_text += "**Recommendation Rationale:**\n"
#         formatted_text += "These products were selected for their excellent features and competitive pricing."
#     state["messages"] = [formatted_text]
#     state["followup_type"] = "selection"
#     state["extraction_completed"] = True
#     state["formatted_text"] = formatted_text
#     return state

class TelecomPlan(BaseModel):
    Provider: str = Field(description="Name of the telecom provider")
    Plan_Details: str = Field(description="Details of the telecom plan")
    Price: str = Field(description="Price of the plan")
    Observations: str = Field(description="Key observations about the plan")
    Recommended_Action_for_Econet: str = Field(description="Recommended action for Econet based on this plan")
    Best_Suited_Econet_Plan_Equivalent: str = Field(description="Best suited equivalent Econet plan")

class TelecomPlans(BaseModel):
    plans: List[TelecomPlan]
# def aggregate_results(state: GraphState) -> GraphState:
#     """Aggregate and rank results based on quality scores."""
#
#
#     return {"aggregated_results": aggregated_results}


def aggregate_results(state: GraphState):
    print("aggregate_results")
    return state

def build_graph():
    # Create the graph
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("router", router_node)  # Use router_node as the node function
    workflow.add_node("search", search_with_tavily)
    workflow.add_node("follow_up", follow_up)
    workflow.add_node("extract_content", extract_content)
    workflow.add_node("aggregate_results", aggregate_results)
    workflow.add_node("format_recommendations", format_recommendations)

    workflow.add_edge(START, "router")  # Start now goes to router

    # Conditional edges from router using router_condition
    workflow.add_conditional_edges(
        "router",
        router_condition,  # Use router_condition for conditional logic
        ["search", "follow_up"]
    )

    # Original edges
    workflow.add_edge("search", "extract_content")
    # workflow.add_conditional_edges("search", route_to_extraction, ["extract_content"])
    workflow.add_edge("extract_content", "aggregate_results")
    workflow.add_edge("aggregate_results", "format_recommendations")

    # Connect follow_up to the appropriate next step
    workflow.add_edge("follow_up", END)  # Follow-up goes straight to recommendations

    workflow.add_edge("format_recommendations", END)

    # Compile the graph
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)
    return graph


graph = build_graph()
thread_config = {"configurable": {"thread_id": "1234"}}
# query = " get the telecel Zimbabwe latest products "
# result = graph.invoke({"query": query,
#     "extraction_completed": False },config=thread_config)
#
#
#
# query = " i will select the first option "
# result = graph.invoke({"query": query,
#     "extraction_completed": True },config=thread_config)
# # #
# # #
# # #
# query = " i confrm lets proceed "
# result = graph.invoke({"query": query,
#     "extraction_completed": True },config=thread_config)
