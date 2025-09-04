import networkx as nx
import json
import pickle
from typing import Dict, List, Tuple
import asyncio
import sys
# 假设import您的模块
from response import *  # 包含openai_response, client (for embedding), openai_response_sync if needed
from .reasoning_player import AgentPlayer
import re
# 假设INQUIRY_COT是常量，如"Reason step-by-step."
INQUIRY_COT= "According to the game expert's prediction, think carefully about your next step of strategy to be most likely to win, and finally make a decision."
GAME_SETTING = ("SURVIVE CHALLENGE!!! You are one of five players in the game. "
                "Everyone is required to choose an integer between 1 and 100 in each round. "
                "The player whose chosen number is closest to (0.8 * the average of all chosen numbers) wins the round without any HP deduction. All other players will have 1 HP deducted. "
                "But if all players choose the same number, their health points are deducted together.")

INQUIRY_COT = ("Ok, {name}! Now is the ROUND {round}, and your HP is at {hp}. "
                "Guess which number will win in the next round. Let's think step by step, and finally answer a number you think you can win.")

class CROMAgent(AgentPlayer):
    def __init__(self, name, persona, engine="gpt-4o-mini", prev_biddings="",
                 summary_model="gpt-4o-mini", external_knowledge_api=None, observation_vars: List[str] = None, 
                 K=2, M=3, mode='load', build_graph=False, background="", state_path='crom_state.pkl'):
        super().__init__(name, persona, engine,"")
        self.engine = engine
        self.summary_model = summary_model
        self.external_knowledge_api = external_knowledge_api
        self.background_rules = GAME_SETTING + f"You are {self.name}"+' The names of five players are Alex, Bob, Cindy, David and Eric.'

        self.per_opp_vars = ["previous_choice", "historical_choices"] # 默认per-opp变量，适应您的建议
        self.shared_observation_vars = ["round", "previous_best_number", "all_hps", "other_players_previous_choices"] 
        self.private_observation_vars =  [] 
        self.observation_vars = self.shared_observation_vars + self.private_observation_vars + self.per_opp_vars 
        self.graph = self._initialize_graph(self.observation_vars)
        self.node_counts = {v: 0 for v in self.graph.nodes if v != 'action'}
        self.node_descriptions = {v: f"Observation {v}" for v in self.observation_vars}
        self.node_descriptions['action'] = "Opponent's action"
        self.example_pool = []  # 共享池
        self.history = []  # list of (obs_dict, pred_action, actual_action)
        self.actions = []  # 新增：记录自己的行动，参考parse_llm_output
        self.K = K
        self.M = M
        self.build_graph = build_graph
        self.state_path = state_path
        all_players = ['Alex', 'Bob', 'Cindy', 'David', 'Eric']  # 假设固定
        self.opponent_ids = sorted([p for p in all_players if p != self.name])  # e.g., 如果self.name='Alex'，opponent_ids=['Bob', 'Cindy', 'David', 'Eric']
        self.name_to_id = {name: f"opponent{i+1}" for i, name in enumerate(self.opponent_ids)}  # 映射：'Bob' -> 'opponent1' 等
        self.current_round_info = {}  # 新增：存储notice_round_result数据
        self.history = []  # 修改为 list of (obs_dict, predicted_actions_dict, actual_actions_dict)
        self.last_inference_trajectories = {}  # 修改为 {opponent_id: trajectory}
        self.first=True
        custom_descriptions = {
        'round': "round",
        'previous_best_number': "The winning number from the previous round",
        'all_hps': "List of all players' HP",
        'other_players_previous_choices': "list of other players' choices in last round.",
        'previous_choice': "This opponent's own choice in the previous round",
        'historical_choices': "This opponent's full history of choices across all previous rounds",
        'action': "This opponent's current action."
         }
        self.node_descriptions.update(custom_descriptions) 
        if mode == 'load':
            self.load_state()

    # ... (保持 _initialize_graph, save_state, load_state 同上一版)

    def _initialize_graph(self, shared_vars):
        graph = nx.DiGraph()
        graph.add_node('action')
        for var in shared_vars + self.per_opp_vars:  # per_opp也加初始边
            graph.add_node(var)
        for var in self.private_observation_vars:
            graph.add_node(var)  # 无边
        graph.add_edge("previous_choice", "action")
        return graph

    def save_state(self):
        state = {
            'graph': self.graph,
            'node_counts': self.node_counts,
            'node_descriptions': self.node_descriptions,
            'example_pool': self.example_pool,

        }
        with open(self.state_path, 'wb') as f:
            pickle.dump(state, f)

    def load_state(self):
        try:
            with open(self.state_path, 'rb') as f:
                state = pickle.load(f)
            self.graph = state['graph']
            self.node_counts = state['node_counts']
            self.node_descriptions = state['node_descriptions']
            self.example_pool = state['example_pool']
            self.first=False
        except FileNotFoundError:
            print("No state file found, initializing new.")

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


    async def MDP_act(self, input_text: str) -> str:
        parsed = await self.parse_input(input_text)
        observation = parsed.get('observation', {})
        actual_opponent_actions = parsed.get('actual_opponent_actions', {})  # dict {opponent_id: action}

        if actual_opponent_actions and self.history:
            last_obs, last_preds, _ = self.history[-1]
            await self.update(last_obs, actual_opponent_actions, last_preds)

        predicted_actions, inference_trajectories = await self.predict_opponent_action(observation)  # 返回 dicts
        self.last_inference_trajectories = inference_trajectories

        my_action = await self._decide_my_action(input_text, predicted_actions)  # 修改为接收dict
        
        self.actions.append(my_action)

        self.history.append((observation, predicted_actions, None))
        self.save_state()

        return my_action
    
    def notice_round_result(self, round, bidding_info, round_target, win, bidding_details, history_biddings):
        super().notice_round_result(round, bidding_info, round_target, win, bidding_details, history_biddings)
  
        
        # 更新current_round_info，用于parse_input
        self.current_round_info = {
            'round': round,
            'bidding_info': bidding_info,  # 字符串
            'bidding_details': bidding_details,  # 字符串
            'history_biddings': history_biddings,  # dict
            'round_target': round_target,  # 0.8*平均
            'win': win  # 获胜者（但我们匿名化）
        }


    async def parse_input(self, input_text: str) -> Dict:  # 保持async以兼容，但内部同步
        if self.first:
            self.first = False
            return {"observation": {}, "actual_opponent_actions": {}, "per_opp_observations": {}}

        info = self.current_round_info  # 从类成员获取结构化数据
        if not info:
            return {"observation": {}, "actual_opponent_actions": {}, "per_opp_observations": {}}

        # 提取共享观察
        observation = {
            'round': info['round'],
            'previous_best_number': info['round_target'],
            'all_hps': self._extract_all_hps(info['bidding_info']),  # 从字符串解析匿名HP list
        }
        # 提取 per_opp_observations 和 actual_opponent_actions（假设 history_biddings 包括最新轮，bids[-1] 是 actual action）
        actual_opponent_actions = {}
        per_opp_observations = {}
        history_biddings = info['history_biddings']  # dict {'Alex': [38, 42], ...} 假设到最新轮

        for name, bids in history_biddings.items():
            if name == self.name:
                continue  # 跳过己方
            opp_id = self.name_to_id.get(name)  # 匿名id，如'opponent1'
            if opp_id and bids:  # 确保有历史
                historical_choices = bids  # 全历史列表
                previous_choice = bids[-2] if len(bids) >= 2 else None  # 上一轮（-2 因为 -1 是当前轮？假设 notice_round_result 在轮后调用，history_biddings[-1] 是上一轮实际）
                actual_action = bids[-1]  # 最新轮的实际动作（用于 
                per_opp_observations[opp_id] = {
                    'historical_choices': historical_choices,
                    'previous_choice': previous_choice
                }
                actual_opponent_actions[opp_id] = actual_action  # 填充 actual

        # 计算共享 'other_players_previous_choices' 作为所有对手上一轮平均（全局视图，用于 observation）
        all_opp_previous = [prev for opp in per_opp_observations.values() if (prev := opp.get('previous_choice')) is not None]
        observation['other_players_previous_choices'] = sum(all_opp_previous) / len(all_opp_previous) if all_opp_previous else None

        observation['per_opp'] = per_opp_observations
        return {"observation": observation, "actual_opponent_actions": actual_opponent_actions}

    async def predict_opponent_action(self, observation: Dict[str, any]) -> Tuple[Dict[str, str], Dict[str, List[Tuple]]]:
        per_opp_obs = observation.get('per_opp', {})
        if per_opp_obs == {}:
            return {}, {}  # 修改为返回空 dict（原是 []，但类型是 Dict）

        # 新增：Semaphore 限并发（从之前响应，如果已添加；否则忽略）

        async def process_predict_opp(opp_name):  # opp_name 是真实名，如 'Bob'
            # 获取匿名 opp_id
            opp_id = self.name_to_id.get(opp_name, opp_name)  # fallback 到 opp_name 如果无映射
            self.print_overwrite(f"Starting prediction {opp_id} (for {opp_name})")  # 可选：打印匿名 + 真实以调试
            
            current_opp_obs = per_opp_obs.get(opp_id, {var: None for var in self.per_opp_vars})
            opp_observation = {var: observation.get(var, None) for var in self.shared_observation_vars + self.private_observation_vars}
            # 绑定 per_opp_vars
            for var in self.per_opp_vars:
                opp_observation[var] = current_opp_obs.get(var, None)
            
            # 新增：动态绑定 "other_players_previous_choices"（排除当前 opp）
            other_previous = []
            for oid in self.opponent_ids:  # oid 是真实名
                if oid == opp_name:  # 用真实名比较
                    continue
                # 获取 oid 的匿名 id，并访问 per_opp_obs
                oid_anon = self.name_to_id.get(oid, oid)
                prev_choice = per_opp_obs.get(oid_anon, {}).get('previous_choice')
                if prev_choice is not None:
                    other_previous.append(prev_choice)
            opp_observation['other_players_previous_choices'] = sum(other_previous) / len(other_previous) if other_previous else None

            topo_order = list(nx.topological_sort(self.graph))
            node_values = {var: opp_observation.get(var, None) for var in self.observation_vars}
            trajectory = []
            for node in topo_order:
                if node in self.observation_vars:
                    continue
                parents = list(self.graph.predecessors(node))
                parent_values = {p: node_values[p] for p in parents}
                value, reasoning = await self._infer_node_value(node, parent_values, opp_id, opp_observation)  # 用 opp_id (匿名)
                node_values[node] = value
                trajectory.append((node, value, reasoning, parents))

            predicted_action = node_values.get('action', 'unknown')
            return opp_id, predicted_action, trajectory  # 返回匿名 opp_id + 值

        # 创建任务列表（用真实名 self.opponent_ids）
        tasks = [process_predict_opp(opp_name) for opp_name in self.opponent_ids]

        # 并行执行（从之前响应，如果已添加；否则用串行 for）
        results = await asyncio.gather(*tasks)

        # 填充输出 dict（用匿名 opp_id 作为 key）
        predicted_actions = {}
        inference_trajectories = {}
        for opp_id, predicted_action, trajectory in results:
            predicted_actions[opp_id] = predicted_action
            inference_trajectories[opp_id] = trajectory

        return predicted_actions, inference_trajectories


    def _extract_all_hps(self, bidding_info: str) -> List[int]:
        # 简单字符串解析：假设格式"NAME:Alex\tHEALTH POINT:10, ..."，提取数字
        hp_strs = [
            part.split(':')[-1].strip().rstrip('.')
            for part in bidding_info.split('After the deduction, player information is: ')[-1].split(', ')
            if 'HEALTH POINT' in part
        ]
        return [int(hp) for hp in hp_strs]  # 假设顺序固定：Alex, Bob, ... -> [10,9,9,9,9]

    # 新增辅助：获取my_hp（基于self.name的固定索引，假设all_hps顺序是all_players）
    def _get_my_hp(self, all_hps: List[int]) -> int:
        all_players = ['Alex', 'Bob', 'Cindy', 'David', 'Eric']
        my_index = all_players.index(self.name)
        return all_hps[my_index] if my_index < len(all_hps) else None


    async def _infer_node_value(self, node: str, parent_values: Dict, opp_id: str = None, opp_observation: Dict = None) -> Tuple[str, str]:
        self.print_overwrite(f"Starting inference for node: '{node}'...")
        query = f"Opponent {opp_id}: Parents: {json.dumps(parent_values)}"
        examples = await self._retrieve_examples(query)
        private_info = ""
        if opp_observation:
            private_parts = {var: opp_observation[var] for var in self.private_observation_vars if opp_observation.get(var)}
            if private_parts:
                private_info = f"Consider your private information: {json.dumps(private_parts)}. Use it to inform reasoning, but do not assume it directly affects the opponent."
        inquiry = f"""
        Your task is to infer the value of the node '{node}' (description: {self.node_descriptions[node]}) 
        based on its parent nodes and their values: {json.dumps(parent_values)}. 
        {private_info}  
        Use the following similar examples for guidance: {examples}. 
        Reason step-by-step about how the parents influence this node, then output the inferred value.
        Output in JSON: {{'value': 'inferred_value', 'reasoning': 'your step-by-step text'}}
        """
        messages = self.construct_prompt(
            last_step_result=None,
            step_and_task="Infer node value",
            external_knowledge=examples,
            inquiry=inquiry
        )
        parsed = await self.call_and_extract_with_retry(messages, self.engine)
        return parsed.get('value', 'unknown'), parsed.get('reasoning', '')
    


    async def _retrieve_examples(self, query: str) -> List[Dict]:
        if not self.example_pool:
            return []
        query_emb = self.get_embedding(query)
        example_embs = [self.get_embedding(e['parent_values']) for e in self.example_pool]
        similarities = [self.cosine_similarity(query_emb, emb) for emb in example_embs]
        top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:self.M]
        return [self.example_pool[i] for i in top_indices]

    def get_embedding(self, text: str) -> List[float]:
        response = client.embeddings.create(input=text, model="text-embedding-ada-002")  # 用response的client
        return response.data[0].embedding

    @staticmethod
    def cosine_similarity(emb1: List[float], emb2: List[float]) -> float:
        from numpy import dot
        from numpy.linalg import norm
        return dot(emb1, emb2) / (norm(emb1) * norm(emb2))

    async def update(self, observation: Dict, actual_opponent_actions: Dict[str, str], predicted_actions: Dict[str, str]):
        per_opp_obs = observation.get('per_opp', {})
        if predicted_actions == []:
            predicted_actions = {"opponent1": "40", "opponent2": "40", "opponent3": "40", "opponent4": "40"}

        # 新增：Semaphore 限并发（可选，防止 LLM API 限流；设为2表示最多2个并发）
        semaphore = asyncio.Semaphore(4)

        async def process_opp(opp_id, actual_action):
            async with semaphore:  # 限流
                try:
                    predicted_action = predicted_actions.get(opp_id)
                    is_correct = await self._is_prediction_correct(predicted_action, actual_action)
                    print(f"{is_correct} for {opp_id} predicted_action：{predicted_action}； actual_action：{actual_action}")

                    examples_to_add = []  # 收集本地 examples
                    if is_correct and opp_id in self.last_inference_trajectories:  # 修复：直接用 opp_id 检查（假设 trajectories key 是 opp_id 如 'opponent1'）
                        print("example_pool append for {opp_id}")
                        trajectory = self.last_inference_trajectories[opp_id]
                        for node, value, reasoning, parents in trajectory:
                            parent_str = json.dumps({p: observation.get(p, None) for p in parents})
                            example = {
                                'parent_values': parent_str,
                                'child_value': value,
                                'reasoning_text': reasoning,
                                'target_link': node,
                                'opponent_id': opp_id
                            }
                            examples_to_add.append(example)  # 收集，不直接 append 到 pool

                    causal_chains = []  # 收集 chains
                    if self.build_graph:
                        current_opp_obs = per_opp_obs.get(opp_id, {})
                        other_context = {oid: per_opp.get('previous_choice') for oid, per_opp in per_opp_obs.items() if oid != opp_id}
                        other_context_str = json.dumps(other_context)
                        reflection_text = await self._reflect(observation, actual_action, current_opp_obs, other_context_str)
                        causal_chains = await self._extract_causal_chains(reflection_text)
                        # 注意：不在这里调用 _update_graph，收集 chains 后统一处理

                    return examples_to_add, causal_chains  # 返回结果
                except Exception as e:
                    print(f"Error processing {opp_id}: {e}")
                    return [], []  # 异常时返回空，避免整体失败

        # 创建任务列表
        tasks = [process_opp(opp_id, actual_action) for opp_id, actual_action in actual_opponent_actions.items()]

        # 并行执行
        results = await asyncio.gather(*tasks)

        # 串行处理结果：更新共享变量
        for examples_to_add, causal_chains in results:
            self.example_pool.extend(examples_to_add)  # 安全 append（extend 以保持顺序）
            if self.build_graph and causal_chains:
                await self._update_graph(causal_chains)  # 串行更新 graph（避免并发修改）

        if self.build_graph:
            self._prune_graph()  # 统一 prune

        if self.history:
            self.history[-1] = (self.history[-1][0], self.history[-1][1], actual_opponent_actions)

    async def _is_prediction_correct(self, predicted: str, actual: str) -> bool:
        sys_prompt1 = f"You are a game expert analyzing player's actions. Background rule of game: {GAME_SETTING}"
        sys_prompt2 = """Understand the game rules and evaluate if the predicted action matches the actual action(within 20% deviation).
        If it reasonably corresponds, answer "PRED 1 X" at end, X is deviation percentage (0 if unknown).
        If not, answer "PRED 0 X" at end, X is deviation percentage (0 if unknown)."""
        inquiry = f"Predicted action: {predicted}. Actual action: {actual}."
        prompt = self.construct_prompt(None, "Evaluate prediction", None, sys_prompt1 + "\n" + sys_prompt2 + "\n" + inquiry)
        response = await self.call_llm(prompt, self.engine)
        if "PRED 1" in response :  # 解析X
            return True
        return False

    async def _reflect(self, observation: Dict, actual_action: str, current_opp_obs: Dict, other_context_str: str) -> str:
        self.print_overwrite("Starting reflection on interaction...")

        shared_obs = {var: observation.get(var) for var in self.shared_observation_vars + self.private_observation_vars}
        # 新增：显式包括other_players_previous_choices
        other_choices_val = shared_obs.get('other_players_previous_choices', 'None')
        
        # 新增：提取相关节点的描述（从 self.node_descriptions）
        relevant_nodes = set(shared_obs.keys()) | set(current_opp_obs.keys()) | {'action', 'action'}  # 包括核心节点
        node_descriptions = {node: self.node_descriptions.get(node, " ") for node in relevant_nodes}
        
        inquiry = f"""
        Your task is to reflect on opponent's action.
        Game rule :{GAME_SETTING}
        Node Descriptions: {json.dumps(node_descriptions)} (use these to understand the meaning of each observation variable).
        
        The value of Node are following:
        Shared observation: {json.dumps(shared_obs)} (including other_players_previous_choices: {other_choices_val} as average of others' previous choices),
        Current opponent's specific obs: {json.dumps(current_opp_obs)},
        Other opponents' context: {other_context_str}, This opponent's action: {actual_action}.
        Analyze this action for the current opponent to see if it reveals any new strategies or beliefs. 
        Focus on how its own previous_choice and other_players_previous_choices influence its action.
        Hypothesize causal links in natural language, using general node names like 'other_players_previous_choices' -> 'average' -> 'action'.
        """
        
        prompt = self.construct_prompt(
            last_step_result=None,
            step_and_task="Reflect on interaction",
            external_knowledge=self.history,
            inquiry=inquiry
        )
        
        self.print_overwrite("Reflecting on interaction... (Calling LLM)")
        reflection_text = await self.call_llm(prompt, self.engine)
        
        self.print_overwrite("Completed reflection.          ")
        return reflection_text

    async def _extract_causal_chains(self, reflection_text: str) -> List[List[str]]:
        self.print_overwrite("Starting extraction of causal chains...")

        # 提供可能的观察变量作为 external_knowledge，帮助 LLM 标准化起始节点
        known_observation_vars = self.observation_vars  # e.g., ['round', 'previous_best_number', ..., 'previous_choice', 'historical_choices']

        # 构建更详细的 inquiry，添加禁止 Markdown 的指令
        inquiry = f"""
        Your task is to parse the following reflection text and extract causal chains that explain the opponent's action.
        A causal chain is a sequence starting from an observation variable, optionally through ONE intermediate node, and ending with 'action'.

        Guidelines:
        - Start with known observation variables: {known_observation_vars}. Use simplified terms from text for nodes.
        - Chains must be 2 or 3 nodes: [obs, 'action'] or [obs, intermediate, 'action']. Keep simple; no sub-decomposition or extra reflection.
        - Extract at most 1 chains; prioritize most prominent in text.
        - Limit to at most 1 unique intermediate node across all chains.
        - If no clear chains fit, output an empty list [].

        Example 1: Text "previous_choice influences action" → {{"chains": [["previous_choice", "action"]]}}
        Example 2: Text "other_players_previous_choices -> average -> action; historical_choices -> action" → {{"chains": [["other_players_previous_choices", "average", "action"], ["historical_choices", "action"]]}}

        Now, parse this reflection text: {reflection_text}

        Output ONLY a JSON object like {{"chains": [["previous_choice", "action"], ["other_players_previous_choices", "average", "action"]]}}. No extra text or code blocks.        """

        messages = self.construct_prompt(
            last_step_result=reflection_text,  # 已包含在 inquiry 中，但保持兼容
            step_and_task="Extract causal chains from reflection",
            external_knowledge=known_observation_vars,  # 传入变量列表作为额外知识
            inquiry=inquiry
        )
        extracted = await self.call_and_extract_with_retry(messages, "gpt-4o")
        extracted = extracted.get('chains', [])
        # 后处理：确保输出是 List[List[str]]，并将结束节点标准化为 'action'（如果需要）
        if isinstance(extracted, list) and all(isinstance(chain, list) and all(isinstance(node, str) for node in chain) for chain in extracted):
            for chain in extracted:
                if chain and chain[-1] != 'action':
                    chain[-1] = 'action'  # 强制标准化（可选，如果文本中用了其他词）
            return extracted
        else:
            print("Invalid extraction, returning empty list")
            return []  # Fallback 如果解析失败

    async def _update_graph(self, causal_chains: List[List[str]]):
        for chain in causal_chains:
            for i in range(len(chain) - 1):
                parent, child = chain[i], chain[i+1]
                match = await self._match_node(child, list(self.node_descriptions.keys())) #match 这里的逻辑不对，相似的不能直接赋值
                if match and self.graph.has_node(match):
                    child = match
                    self.node_counts[child] += 1
                else:
                    desc = await self._generate_node_description(child, chain)
                    self.graph.add_node(child)
                    self.node_descriptions[child] = desc
                    self.node_counts[child] = 1
                self.graph.add_edge(parent, child)
                if not nx.is_directed_acyclic_graph(self.graph):
                    self.graph.remove_edge(parent, child)  # 如果引入环，移除
                    
                    print(f"Removed edge {parent} -> {child} to avoid cycle.")
        self.save_state()

    async def _match_node(self, new_node: str, existing_nodes: List[str]) -> str or None:
        self.print_overwrite(f"Starting node matching: '{new_node}'...")
        
        # 新增：准备现有节点的描述映射（假设 self.node_descriptions 存在）
        existing_descriptions = {node: self.node_descriptions.get(node, " ") for node in existing_nodes}
        
        # 新增：游戏背景（简述 SURVIVE CHALLENGE 规则）
        game_background = GAME_SETTING
        
        # 改进 inquiry：添加语义匹配、描述、背景、few-shot 示例
        inquiry = f"""
        {game_background}
        
        Your task is to determine if the new node '{new_node}' is semantically equivalent to any existing node in the list.
        - "Semantically equivalent" means the nodes represent the same concept in the game context, even if names differ slightly (e.g., 'perceived_avg' matches 'perceived_average' if they both mean the opponent's estimated average of others' choices).
        - Consider node descriptions: {json.dumps(existing_descriptions)}.
        - If there's a match, output the exact name of the matching existing node.
        - If no match, output null.

        Examples:
        - New: 'avg', Existing: ['average' (desc: 'Estimated average of opponents' choices')], Output: {{"match": "average"}}
        - New: 'risk', Existing: ['strategy' (desc: 'Overall plan')], Output: {{"match": null}} (not equivalent)

        Output ONLY the JSON object: {{"match": "node_name"}} or {{"match": null}}. Use double quotes, no extra text.
        """
        
        messages = self.construct_prompt(
            last_step_result=None,
            step_and_task="Match or add node",
            external_knowledge=existing_nodes,  # 保留，但 inquiry 已包含描述
            inquiry=inquiry
        )
        
        self.print_overwrite(f"Matching node '{new_node}'... (Calling LLM)")
        parsed = await self.call_and_extract_with_retry(messages, self.engine)
        
        self.print_overwrite("Completed node matching.      ")
        
        match_value = parsed.get('match')
        return match_value if match_value is not None else None  # 处理 null 为 None

    async def _generate_node_description(self, new_node: str, chain: List[str]) -> str:
        messages = self.construct_prompt(
            last_step_result=None,
            step_and_task="Generate node description",
            external_knowledge=chain,
            inquiry=f"For new node '{new_node}' in chain {chain}, generate a concise description. Output JSON: {{'description': 'text'}}"
        )
        parsed = await self.call_and_extract_with_retry(messages, self.engine)
        return parsed.get('description', f"Intermediate: {new_node}")

    def _prune_graph(self):
        intermediate_nodes = [n for n in self.graph.nodes if n not in self.observation_vars + ['action']]
        sorted_nodes = sorted(intermediate_nodes, key=lambda n: self.node_counts[n], reverse=True)
        to_keep = sorted_nodes[:self.K]
        to_prune = set(intermediate_nodes) - set(to_keep)
        print(f"node_counts... {self.node_counts}")
        for node in to_prune:
            self.graph.remove_node(node)
            del self.node_counts[node]
            del self.node_descriptions[node]

    async def _decide_my_action(self, raw_observation: str, predicted_actions: Dict[str, str]) -> str:
        predicted_str = json.dumps(predicted_actions)  # 或总结为"Predicted actions of opponents: {predicted_str}"
        inquiry = f"Expert predictions of opponents' actions: {predicted_str}. Output your strategy as string."

        
        messages=[{'role': 'system', 'content': GAME_SETTING}]+ raw_observation+ [ {'role': 'system', 'content': inquiry}]
        #messages+=new_message
        self.llm_response = await self.call_llm(messages, self.engine)
        action = await self.parse_llm_output(self.llm_response)
        return action

    @async_adapter
    async def parse_llm_output(self, llm_response,record=True):
        """
        将LLM的输出解析为可执行动作
        """
        action= await self.parse_result(llm_response)
        if record:
            self.biddings.append(action)
        return action
    
    # 接口方法实现
    async def call_llm(self, messages: List[Dict], model: str) -> str:
        # 参考提供代码，添加参数
        return await openai_response(
            model=model,
            messages=messages,
            max_tokens=800,
            temperature=0.7,
            top_p=0.9,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )

    def construct_prompt(self, last_step_result, step_and_task, external_knowledge, inquiry=INQUIRY_COT):
        """参考提供代码，构建messages列表"""
        if isinstance(last_step_result, list):
            messages = [
                {'role': 'system', 'content': "The following is game record:"},
            ]
            for prompt in last_step_result:
                messages.append({'role': 'user', 'content': prompt})
        else:
            messages = [
                {'role': 'system', 'content': f"The following is game record: {last_step_result or ''}"},
            ]
        ex_prompt = ""
        if external_knowledge:
            ex_prompt += f"An game expert predict other players' strategies are: {external_knowledge}.\n "
        user_prompt = f"OK, {self.name}! " + step_and_task + ex_prompt + inquiry
        messages.append({'role': 'system', 'content': user_prompt})
        return messages

    
    def extract_json_from_text(self, text: str) -> Dict:  # 返回类型改为 Dict（强制 {}）
        # 步骤1: 清理多余符号
        text = text.strip()
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'\s*```', '', text)
        text = re.sub(r'^\s*\n|\n\s*$', '', text)

        # 新增：如果看起来是单引号 JSON，替换为双引号（处理常见问题）
        if "'" in text and '"' not in text:
            text = text.replace("'", '"')

        # 步骤2: 优先尝试直接解析整个文本（假设它是 {}）
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed  # 成功，且是 {}
            else:
                raise json.JSONDecodeError("Not a dict", text, 0)  # 如果是 [] 或其他，强制失败
        except json.JSONDecodeError:
            pass

        # 步骤3: Fallback - 使用平衡括号提取 ONLY {} 结构（忽略 []）
        def find_json_object(s: str) -> str:
            if not s:
                return ""
            # 找到起始 '{'
            start = -1
            for i, char in enumerate(s):
                if char == '{':
                    start = i
                    break
            if start == -1:
                return ""  # 没有 {}，返回空

            # 计数平衡括号，只匹配 {}
            balance = 0
            for i in range(start, len(s)):
                if s[i] == '{':
                    balance += 1
                elif s[i] == '}':
                    balance -= 1
                if balance == 0:
                    return s[start:i+1]
            return ""  # 不平衡

        json_data = find_json_object(text)

        # 步骤4: 尝试解析提取的部分
        if json_data:
            try:
                parsed = json.loads(json_data)
                if isinstance(parsed, dict):
                    return parsed
                else:
                    raise json.JSONDecodeError("Extracted structure is not a dict", json_data, 0)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON after extraction: {e}")
        else:
            raise ValueError("No valid {} JSON object found in response")
        
    async def call_and_extract_with_retry(self, messages: List[Dict], model: str, max_retries=2) -> Dict or List:
        """重试helper：循环调用call_llm + extract，直到成功或max_retries"""
        n_retry = 0
        while n_retry < max_retries:
            try:
                response = await self.call_llm(messages, model)
                js_dict = self.extract_json_from_text(response)
                return js_dict
            except Exception as e:  # 广义捕获，包括call_llm异常和extract错误
                n_retry += 1
                print(f"Retry {n_retry}: Failed ({e}), retrying...")
        print(f"Max retries reached, returning empty dict")
        return {}  # 
    
    def print_overwrite(self, message: str):
        sys.stdout.write(f"\r[Progress] {message}")
        sys.stdout.flush()