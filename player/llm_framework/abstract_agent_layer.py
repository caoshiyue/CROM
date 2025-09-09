##
# Author:  
# Description:  
# LastEditors: Shiyuec
# LastEditTime: 2025-09-09 07:25:47
## 
from abc import ABC, abstractmethod
from typing import Any, Dict
from response import openai_response

class AbstractAgent(ABC):
    """
    抽象Agent接口：输入文本，输出动作
    """

    def __init__(self, config, **kwargs):
        self.config = config
        self.model= kwargs["model"]

    async def act(self, input_text) -> Dict[str, Any]:
        """
        接收环境文本，返回结构化动作。
        核心流程：
        1. 调用子类实现的 get_action
        2. 调用 config.parse_action 做解析
        """
        raw_action = await self.get_action(input_text)
        parsed_action = self.config.parse_action(raw_action)
        return parsed_action

    @abstractmethod
    async def get_action(self, input_text) -> str:
        """
        子类实现：如何根据输入文本生成原始动作（通常是LLM回复 / 规则输出）
        必须返回字符串（raw_action）
        """
        pass


class LLMOnly(AbstractAgent):
    """
    一个最小的 LLM Agent 示例
    """

    async def get_action(self, input_text) -> str:
        # 这里只负责生成原始回复
        res = await openai_response(
            model = self.model,
            messages=input_text,
            temperature=0.7,
            max_tokens=1600,
            )
        
        return res

class Cot(AbstractAgent):

    async def get_action(self, input_text) -> str:
        input_text+=[{"role":"system","content": self.config.get_prompt("cot")}]
        res = await openai_response(
            model = self.model,
            messages=input_text,
            temperature=0.7,
            max_tokens=1600,
            )
        
        return res
    
class Tot(AbstractAgent):

    async def get_action(self, input_text) -> str:
        input_text+=[{"role":"system","content": self.config.get_prompt("tot")}]
        res = await openai_response(
            model = self.model,
            messages=input_text,
            temperature=0.7,
            max_tokens=1600,
            )
        
        return res




class RuleBasedAgent(AbstractAgent):
    """
    ! 测试用：
    - 首先用 config.parse_state 解析环境文本
    - 然后基于 state 做一个简单决策
    """

    async def get_action(self, input_text) -> str:
        # 解析 state
        state = await self.config.parse_state(input_text)
        print(state)
        # 简单决策逻辑
        target = state.get("target", 50) or 50

        choice = int(target*0.8)
        return f"I choose {choice}"