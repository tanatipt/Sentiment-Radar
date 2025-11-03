from pydantic import BaseModel , Field
from typing_extensions import List, Literal, Annotated, Dict
from langchain.docstore.document import Document
from langchain_core.messages import BaseMessage
from typing_extensions import Optional
from langgraph.graph.message import add_messages


class AssetInformation(BaseModel):
    """Asset Information"""
    asset_type : Literal["cryptocurrency", "stocks"]
    trading_symbol : str
    trading_exchange: Literal["BINANCE", "NASDAQ"]
    symbol_alias : str

class ModelConfig(BaseModel):
    """Model configuration"""
    model_class : str
    model_params: Dict

class Step(BaseModel):
    """
    A Pydantic model representing a single step in a chain of thought.
    Each step includes a description of the reasoning step and its corresponding output.
    """
    description: str = Field(...,  description="A brief explanation of the reasoning step taken.")
    output: str = Field(...,  description="The result or conclusion derived from this reasoning step.")

class Report(BaseModel):
    """
    A Pydantic model representing a structured market sentiment analysis report
    generated from a list of provided news articles. Do NOT include any citations or references 
    to the news articles within the report itself. (i.e. no in-text citations) All citations must be 
    recorded separately in the `citations` field.
    """

    chain_of_thought: List[Step] = Field(
        [],
        description=(
            "A sequence of reasoning steps representing the structured approach "
            "used to analyze the provided news articles."
        ),
        min_length=1,
    )

    report: str = Field(
        '',
        description=(
            "A detailed analytical report describing the current and, if applicable, "
            "future market sentiment of the asset, strictly grounded in the chain-of-thought reasoning."
        ),
    )

    current_sentiment: Literal[
        'Strongly Negative', 'Negative', 'Neutral', 'Positive', 'Strongly Positive'
    ] = Field(
        '',
        description="The concluded current market sentiment based on the analysis presented in the report.",
    )

    future_sentiment: Optional[
        Literal['Strongly Negative', 'Negative', 'Neutral', 'Positive', 'Strongly Positive']
    ] = Field(
        None,
        description="The projected future market sentiment, if applicable, based on the report.",
    )

    citations: List[int] = Field(
        [],
        description="A list of integer IDs referencing the news articles that support the report’s conclusions.",
    )


class State(BaseModel):
    """
    A Pydantic model representing the workflow state, including retrieved news articles,
    formatted content, and the generated sentiment report.
    """

    messages: Annotated[List[BaseMessage], add_messages] = Field(
        [],
        description="A list of conversation messages exchanged between the generator and critic models.",
    )

    retrieved_news: List[Document] = Field(
        [],
        description="A list of filtered news articles relevant to the current trading symbol.",
    )

    formatted_news: str = Field(
        '',
        description="The filtered news articles formatted into a single text block.",
    )

    self_reflection_passed: bool = Field(
        False,
        description="Indicates whether the generated report successfully passed the self-reflection evaluation.",
    )

    report: Report = Field(
        default_factory=Report,
        description=(
            "The structured sentiment report containing analyses of current and, if applicable, "
            "projected market sentiment, along with supporting citations."
        ),
    )

    email: Optional[str] = Field(
        None,
        description="The sentiment report formatted as an HTML weekly newsletter.",
    )


class GroundednessOutput(BaseModel):
    """
    A Pydantic model for assessing whether a market sentiment report is factually grounded 
    in the news articles it references. 

    If the report is **not factually grounded**, provide a list of specific criticisms 
    explaining how to improve its factual grounding based on the referenced news articles.
    """
    chain_of_thought: List[Step] = Field(
        ...,
        description=(
            "A structured sequence of reasoning steps used to evaluate the factual grounding "
            "of the report with respect to the referenced news articles."
        ),
        min_length=1
    )
    is_grounded: bool = Field(
        ...,
        description=(
            "Indicates whether the report is factually grounded in the referenced articles, "
            "as determined through the chain-of-thought reasoning."
        )
    )
    criticisms: Optional[List[str]] = Field(
        ...,
        description=(
            "If the report is not grounded, provide a list of specific, actionable criticisms "
            "explaining how to make it more factually accurate and aligned with the referenced news articles."
        )
    )


class UsefulnessOutput(BaseModel):
    """
    A Pydantic model for evaluating whether a market sentiment report effectively addresses 
    the current market sentiment and, where relevant, the anticipated future sentiment.

    If the report **does not effectively address** the current or future market sentiment, 
    provide a list of specific criticisms explaining how to improve its usefulness.
    """

    chain_of_thought: List[Step] = Field(
        ...,
        description=(
            "A structured sequence of reasoning steps used to evaluate how well the report "
            "addresses current and potential future market sentiment based on its content."
        ),
        min_length=1
    )
    is_useful: bool = Field(
        ...,
        description=(
            "Indicates whether the report effectively covers the current and, if applicable, "
            "future market sentiment, as determined through the chain-of-thought reasoning."
        )
    )
    criticisms: Optional[List[str]] = Field(
        ...,
        description=(
            "If the report is not useful, provide a list of specific, actionable criticisms "
            "explaining how to make it more relevant and aligned with market sentiment."
        )
    )


