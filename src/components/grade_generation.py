from src.prompts.grade_generation import hallucination_prompt, usefulness_prompt
from src.components.schemas import State, GroundednessOutput, UsefulnessOutput, AssetInformation
from langchain_core.language_models.chat_models import BaseChatModel
from typing_extensions import Literal, List
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from config import settings

def format_criticisms(criticisms: List[str]) -> HumanMessage:
    """
    Format the criticisms into bullet point string

    Args:
        criticisms (List[str]): A list of criticisms
    Returns:
        HumanMessage: A human message of the formatted criticisms
    """
    critcisms_str = f"""
    Criticisms:
    {"\n".join([f"-{critcism}" for critcism in criticisms])}
    """

    return HumanMessage(content = critcisms_str)

def grade_generation(state: State, model : BaseChatModel, asset_information : AssetInformation) -> State:
    """
    Evaluates the generated market sentiment report for groundedness and usefulness.
    Args:
        state (State): The current pipeline state containing the sentiment report and news articles.
        model (BaseChatModel): The language model used for grading the report.
        asset_information (AssetInformation): Information about the trading asset.
    Returns:
        State: An updated state of the graph.
    """
    messages = state.messages
    report = messages[-1].content
    human_msg = """{report}"""

    useful_pt = ChatPromptTemplate(
        [
        ('system', usefulness_prompt),
        ('human', human_msg) 
        ]
    )
    # Create a prompt template for the usefulness evaluation
    useful_chain = useful_pt | model.with_structured_output(UsefulnessOutput)
    # Invoke the model to evaluate the usefulness of the report
    usefulness_response = useful_chain.invoke({"report" : report, 'symbol_alias' : asset_information.symbol_alias})

    if usefulness_response.is_useful:

        hallucination_pt = ChatPromptTemplate(
            [
                ('system', hallucination_prompt),
                ('human', human_msg)
            ]
        )
        # Create a prompt template for the groundedness evaluation
        hallucination_chain = hallucination_pt | model.with_structured_output(GroundednessOutput)
        # Invoke the model to evaluate the groundedness of the report
        groundness_response = hallucination_chain.invoke({"report" : report})

        if groundness_response.is_grounded:
            return {"self_reflection_passed" : True}
        
        return {
            "messages" : [format_criticisms(groundness_response.criticisms)], 
            "self_reflection_passed" : False
        }

    
    return {
        "messages" : [format_criticisms(usefulness_response.criticisms)],
        "self_reflection_passed" : False
    }


def route_flow(state : State) -> Literal["email_formatter", "analyse_sentiment", "__end__"]:
    """
    Routes the next node to visit after self-reflection.

    Args:
        state (State): The current pipeline state containing the sentiment report and news articles.
    Returns:
        Literal["email_formatter", "analyse_sentiment", "__end__"]: Name of the next node to vist.
    """
    if state.self_reflection_passed:
        return "email_formatter"
    
    if len(state.messages) >= settings.max_reflection_round * 2:
        return "__end__"
    
    return "analyse_sentiment"


    
