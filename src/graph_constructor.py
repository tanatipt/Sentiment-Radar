from langgraph.graph import StateGraph, START, END
from src.components.schemas import State, ModelConfig, AssetInformation
from src.components.retrieve_news import retrieve_news
from src.components.analyse_sentiment import analyse_market_sentiment
from src.components.grade_generation import grade_generation, route_flow
from src.components.email_formatter import email_formatter
from src.mapper import get_class
from config import settings
from dotenv import load_dotenv
load_dotenv()

class GraphConstructor:
    def __init__(
        self, 
        generator_config: ModelConfig,
        critic_config : ModelConfig,
        asset_information: AssetInformation
    ):
        """
        Initializes the graph constructor with the necessary parameters for constructing the workflow graph.

        Args:
            generator_config (ModelConfig): Configuration for the generator model.
            critic_config (ModelConfig): Configuration for the critic model.
            asset_information (AssetInformation): Information about the trading asset.
        """
        generator_config = ModelConfig.model_validate(generator_config)
        critic_config = ModelConfig.model_validate(critic_config)
        asset_information = AssetInformation.model_validate(asset_information)

        # Initialize the language models for the workflow
        generator_model = get_class("llm", generator_config.model_class)(**generator_config.model_params)
        critic_model = get_class("llm", critic_config.model_class)(**critic_config.model_params)

        # Initialize the nodes of the workflow with the provided parameters
        self.retrieve_news = self.init_node(retrieve_news, asset_information=asset_information)
        self.analyse_sentiment = self.init_node(analyse_market_sentiment, model = generator_model, asset_information=asset_information)
        self.grade_generation = self.init_node(grade_generation, model = critic_model, asset_information=asset_information)
        self.email_formatter = self.init_node(email_formatter, model = generator_model,  asset_information=asset_information)


        
    def connect_nodes(self) -> StateGraph:
        """
        Connects the nodes of the workflow graph to define the flow of data and control between them.

        Returns:
            StateGraph: A StateGraph object representing the workflow, with nodes connected in a specific order.
        """
        workflow = StateGraph(State)

        workflow.add_node("retrieve_news", self.retrieve_news)
        workflow.add_node("analyse_sentiment", self.analyse_sentiment)
        workflow.add_node('email_formatter', self.email_formatter)
        workflow.add_node("grade_generation", self.grade_generation)

        workflow.add_edge(START, "retrieve_news")
        workflow.add_edge("retrieve_news", "analyse_sentiment")
        workflow.add_edge("analyse_sentiment", "grade_generation")
        workflow.add_conditional_edges("grade_generation", route_flow)
        workflow.add_edge('email_formatter', END)

        return workflow

    def init_node(self, node_function : callable, **kwargs : dict) -> callable:
        """
        Initializes a node function with the provided parameters.

        Args:
            node_function (callable): The function to be wrapped as a node in the workflow.
            **kwargs (dict): Additional keyword arguments to be passed to the node function.

        Returns:
            callable: A wrapped function that takes a State object as input and invokes the original node function with the provided parameters.
        """

        def wrapped_node_function(state : State):
            return node_function(state, **kwargs)
        
        return wrapped_node_function
    
    def compile(self, save_path : str = None) -> StateGraph:
        """
        Compiles the workflow graph by connecting the nodes and returning the final StateGraph object.

        Args:
            save_path (str) : The path to save an image of the workflow graph
        Returns:
            StateGraph: A StateGraph object representing the complete workflow, ready for execution.
        """
        workflow = self.connect_nodes()
        graph = workflow.compile()

        if save_path is not None:
            # Get the graph and draw it as PNG
            png_graph = graph.get_graph().draw_mermaid_png()

            # Save the PNG file
            with open(save_path, "wb") as f:
                f.write(png_graph)

        return graph