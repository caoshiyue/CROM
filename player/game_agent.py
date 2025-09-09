##
# Author:  
# Description:  
# LastEditors: Shiyuec
# LastEditTime: 2025-09-09 02:04:53
##
import os 
from .llm_framework import *
from .reasoning_player import AgentPlayer

# Description 对接部分，按照环境要求修改
class GameAgent(AgentPlayer):
    def __init__(self, agent_type, agent_params=None, config_path=None, **kwargs):
        super().__init__(**kwargs)
        if config_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(base_dir, "llm_framework", "game.yaml")

        self.config = ConfigLoader(config_path)

        if agent_type not in AGENT_REGISTRY:
            raise ValueError(f"Unknown agent_type: {agent_type}. "
                            f"Available: {list(AGENT_REGISTRY.keys())}")

        AgentClass = AGENT_REGISTRY[agent_type]
        self.agent = AgentClass(self.config, **(agent_params or {}))

    async def act(self,):
        ##
        # Description: 重载act,保持输入输出,执行原有act的逻辑
        ## 
        print(f"Player {self.name} conduct bidding ")
        response = await self.agent.act(self.message)
        response = response["res"]
        self.message.append({"role":"assistant","content":response})
        self.biddings.append(await self.parse_result(response))
        


