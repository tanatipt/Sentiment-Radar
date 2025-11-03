from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from typing_extensions import Any

llm_map = {
    "ChatOpenAI": ChatOpenAI,
    "ChatGoogleGenerativeAI" : ChatGoogleGenerativeAI
}



def get_class(map_type : str, name : str) -> Any:
    """
    Returns the appropriate class to instantiate

    Args:
        map_type (str): Mapping type
        name (str): Name of class to retrieve

    Raises:
        Exception: Specified mapping type does not exist
        Exception: Name of class to retrieve from mapping does not exist

    Returns:
        Any: The class to retrieve
    """

    map_dict = {
        "llm" : llm_map,
    }

    if map_type not in map_dict:
        raise Exception('ERROR: Mapping type does not exist')

    map_type_dict = map_dict[map_type]

    if name not in map_type_dict:
        raise Exception('ERROR: Mapping name does not exist in mapping type')
    
    cls = map_type_dict[name]

    return cls