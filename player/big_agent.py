from response import *
import time
import re
import json
from .reasoning_player import AgentPlayer
import asyncio

INQUIRY_COT= "Think carefully about your next step of strategy to be most likely to win. Let's think step by step, and finally make a decision." 
INQUIRY_PERSONA2 = "Don't forget your expert status, use your expertise to win this round!"
INQUIRY_PERSONA1 = "You are a game expert, good at predicting other people's behavior and deducing calculations, and using the most favorable strategy to win the game."
REFLECT_INQUIRY = "Review the previous round games, summarize the experience."
INQUIRY_PCOT= "First of all, predict the next round of choices based on the choices of other players in the previous round."  #需要提供全历史


class Bigagent_(AgentPlayer):
    def __init__(self, name,persona, decision_model="gpt-4o-mini", summary_model="gpt-4o-mini", external_knowledge_api=None):
        super().__init__( name, persona, decision_model)
        """
        父类会执行：
            self.hp = 10
            self.biddings=[]
            self.cur_round = -1
            self.logs = None
            self.engine = decision_model  # 决策LLM模型版本
        """

        self.summary_llm = summary_model  # 总结LLM模型版本
        self.external_knowledge_api = external_knowledge_api
        self.history = []  # 仅记录last_step_result历史
        self.history_prompt = []
        self.background_rules = None  # 背景规则

    @async_adapter
    async def MDP_act(self, input_text):
        """
        主流程：解析输入、构造提示、调用决策LLM、返回动作
        """
        # 如果是首次初始化背景规则，则提取背景规则
        if self.background_rules is None:
            extract_rules = await self.extract_background_rules(input_text)
            self.background_rules = extract_rules["rule"]
            self.persona = extract_rules["persona"]
            self.rule = self.construct_rule(background_rules=self.background_rules,persona = self.persona)
            self.history_prompt = self.rule

        # 1. 使用总结LLM提取关键信息
        parsed_input = await self.parse_input(input_text)

        # 提取输入信息的各部分
        #current_state = parsed_input["current_state"]
        last_step_result = parsed_input["last_step_result"] #! 如果用于 history 是否还需要精简
        step_and_task = parsed_input["step_and_task"]

        # 2. 历史记录更新
        if last_step_result!="None" and last_step_result!=None:
            self.history.append(last_step_result)

        # 3. 外部知识调用（如果需要）
        external_knowledge = ""
        if self.external_knowledge_api:
            external_knowledge = await self.external_knowledge_api.query(
                #current_state=current_state,
                step_and_task=step_and_task
            )
        flat_history_text = "\n ".join([item for item in self.history])
        # 4. 构造LLM请求上下文
        
        prompt = self.construct_prompt(
            last_step_result=last_step_result,
            #current_state=current_state,
            step_and_task=step_and_task,
            external_knowledge=external_knowledge
        )
        
        # 5. 调用决策LLM生成输出，
        self.history_prompt=self.history_prompt+ prompt

        final_prompt= self.history_prompt
        self.llm_response = await self.call_llm(final_prompt, model=self.engine)
        self.history_prompt.append({'role': 'assistant',"content": self.llm_response})

        with open("final_prompt.json", "w", encoding="utf-8") as file:
            json.dump(final_prompt, file, ensure_ascii=False, indent=4)  # 确保中文字符可读，格式化输出
        # 6. 转换LLM输出为可执行动作/决策文本
        action = await self.parse_llm_output(self.llm_response)
        
        return action

    @async_adapter
    async def act(self):   #适配接口
        print(f"Player {self.name} conduct bidding")
        if not hasattr(self, "last_message_length"): # 由于这个game 用append message形式，因此在这里转换成MDP，无需更早轮次信息
            self.last_message_length = 0  
        # 获取新增消息
        new_messages = self.message[self.last_message_length:]
        
        await self.MDP_act(new_messages)
        # self.message需要记录llm response
        self.message.append({'role': 'assistant', 'content': self.llm_response})
        # 更新记录的消息长度
        self.last_message_length = len(self.message)
    
    @async_adapter
    async def parse_input(self, input_text):
        """
        使用总结LLM提取输入文本中的关键信息。
        包含当前状态、上一轮结果、当前步骤要求和Agent任务的合并信息。
        人称的问题，我们需要保证system以第二人称描述
        重复role 的问题，
        goal 和 动作描述的问题
        history 中 人称对不齐的问题
        """

        flat_input_text = "; ".join([f"role: {item['role']}, content: {item['content']}." for item in input_text])
        flat_input_text= self.persona + self.background_rules + flat_input_text
        sys_prompt = self.persona + f"""
        Below is game record related to you. Please extract these information related to game decisions, copying the original wording as much as possible.
        Output the result in JSON format, and ensure the JSON keys match the given schema exactly:
        Schema: 
        - last_step_result: Description of what happened in last round (if not, answer None), including the round number, actions of all players, and outcomes in natural text.
        - step_and_task: Description your acheivement of game and required action in current step in natural text. Retain the second-person perspective.
        """
        #        - current_state: Description of the current game state, you and other players' state in natural text. 
        messages = [
            {'role': 'system', 'content': sys_prompt},
            {'role': 'user', 'content': flat_input_text},
        ]
        n_retry=0
        while n_retry<10:
            try:
                response = await self.call_llm(messages, model=self.summary_llm)
                js_dict=self.extract_json_from_text(response)
                n_retry=10
            except: #再试一次
                n_retry+=1
        return js_dict

    @async_adapter
    async def extract_background_rules(self, input_text):
        """
        提取背景规则, 适用于身份基本不变的game
        """
        flat_input_text = "; ".join([f"role: {item['role']}, content: {item['content']}." for item in input_text])
        sys_prompt = f"""
        The following is a description of a game. Please extract the following information, avoid unnecessary explanations. Output the result in JSON format, and ensure the JSON keys match the given schema exactly:
        - persona: Describe your role, retain the second-person perspective.
        - rule: Description of fixed rules or settings of game in a natural and coherent paragraph.
        """
        messages = [
            {'role': 'system', 'content': sys_prompt},
            {'role': 'user', 'content': flat_input_text},
        ]
        response = await self.call_llm(messages, model=self.summary_llm)
        return self.extract_json_from_text(response)

    def construct_prompt(self, last_step_result,  step_and_task, external_knowledge):
        """
        构造给决策LLM的完整提示上下文
        """
        sys_prompt = f"""
        The following is last step results of game: {last_step_result}
        """
        if external_knowledge:
            sys_prompt+= f"An game expert suggusts {external_knowledge}. \n"

        user_prompt = step_and_task+INQUIRY_COT
        messages = [
            {'role': 'system', 'content': sys_prompt},
            {'role': 'system', 'content': user_prompt},
        ]
        return messages
    def construct_rule(self, background_rules, persona):
        """
        构造给决策LLM的完整提示上下文
        """
        messages = [
            {'role': 'system', 'content': persona + background_rules},
        ]
        return messages

    @async_adapter
    async def call_llm(self, prompt, model):
        """
        调用OpenAI API的通用函数
        """
        response = await openai_response(
            model=model,
            messages=prompt,
            max_tokens=1600,
            temperature=0.7,
            top_p=0.9,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )
        return response

    @async_adapter
    async def parse_llm_output(self, llm_response):
        """
        将LLM的输出解析为可执行动作
        """
        action= await self.parse_result(llm_response)
        self.biddings.append(action)
        return action

    def extract_json_from_text(self, text):
        """
        从LLM返回文本中提取JSON格式的数据
        """
        try:
            json_start = text.index("{")
            json_end = text.rindex("}") + 1
            json_data = text[json_start:json_end]
            return json.loads(json_data)
        except (ValueError, json.JSONDecodeError):
            raise ValueError("未能从LLM响应中提取有效的JSON数据")
