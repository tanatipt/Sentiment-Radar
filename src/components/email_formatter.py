from src.prompts.email_formatter import email_format_prompt
from src.components.schemas import State, AssetInformation
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts.chat import ChatPromptTemplate


def email_formatter(state : State, model:  BaseChatModel, asset_information : AssetInformation) -> State:
    """
    Formats the sentiment report into a structured HTML email newsletter.
    Args:
        state (State): The current pipeline state.
        model (BaseChatModel): The language model used for formatting the content of the email.
        asset_information (AssetInformation): Information about the trading asset.
    Returns:
        State: An updated state of the graph."""

    report = state.messages[-1].content
    human_msg = """{report}"""
    format_pt = ChatPromptTemplate(
        [
            ('system', email_format_prompt),
            ('human', human_msg )
        ]
    )

    format_chain = format_pt | model
    # Invoke the model to format the sentiment report to a HTML newsletter
    email_content = format_chain.invoke(
        {
            "report" : report,
            "symbol_alias" : asset_information.symbol_alias
        }
    ).content

    return {"email" : email_content}