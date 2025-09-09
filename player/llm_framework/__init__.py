##
# Author:  
# Description:  
# LastEditors: Shiyuec
# LastEditTime: 2025-09-09 02:27:34
## 
from .config_layer import ConfigLoader
from .abstract_agent_layer import *
from .utils import extract_json

AGENT_REGISTRY = {
    "llm": LLMOnly,
    "cot": Cot,
    "tot": Tot,
    "rule": RuleBasedAgent,

}