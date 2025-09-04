##
# Author:  
# Description:  
# LastEditors: Shiyuec
# LastEditTime: 2025-09-03 11:01:50
## 
import yaml
import json
import re
from typing import Any, Dict, List
from .utils import extract_json
from response import openai_response


class ConfigLoader:
    def __init__(self, yaml_path: str):
        """
        初始化配置，读取 YAML 文件。
        """
        with open(yaml_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.state_schema = self.config.get("state", {}).get("schema", [])
        self.state_parser_cfg = self.config.get("state", {}).get("parser", {})
        self.action_schema = self.config.get("action", {}).get("schema", [])
        self.action_parser_cfg = self.config.get("action", {}).get("parser", {})
        self.prompts = self.config.get("prompts", {})

        self.state_history: List[Dict[str, Any]] = []


    def get_prompt(self, name: str) -> str:
        """
        获取 YAML 中定义的 prompt。

        Example:
        >>> get_prompt("cot")
        "Let's think step by step..."
        """
        return self.prompts.get(name, "")


    async def parse_state(self, text: str) -> Dict[str, Any]:
        """
        解析环境文本为结构化 state。

        Example:
        >>> await parse_state("Round 3. Player HP: [9,8,7]. Target=46.4")
        {
            "round": 3,
            "hp": [9,8,7],
            "player_choices": [12, 55, 73],
            "target": 46.4
        }
        """
        parser_type = self.state_parser_cfg.get("type")

        if parser_type == "llm":
            model = self.state_parser_cfg.get("model")
            retries = self.state_parser_cfg.get("retries", 1)
            state = await self._parse_state_via_llm(text, model, retries)
        else:
            raise NotImplementedError(f"Unsupported state parser: {parser_type}")

        self.state_history.append(state)
        return state


    async def _parse_state_via_llm(self, text: str, model: str, retries: int) -> Dict[str, Any]:
        """
        使用 LLM 解析环境文本为 JSON。

        Example:
        >>> raw_text = "Round 2. HP=[10,9,8], Choices=[23,45,67], Target=35.2"
        >>> await _parse_state_via_llm(raw_text, "gpt-4o-mini", 2)
        {
            "round": 2,
            "hp": [10,9,8],
            "player_choices": [23,45,67],
            "target": 35.2
        }
        """
        schema_desc = "\n".join(
            [f"- {f['name']} ({f['type']}): {f['description']}" for f in self.state_schema]
        )
        prompt = (
            f"Extract the following fields as JSON from the text:\n{schema_desc}\n\n"
            f"Text:\n{text}"
        )

        for attempt in range(retries):
            resp = await openai_response(model=model, messages=[{"role": "user", "content": prompt}],temperature=0.2,)
            try:
                parsed = extract_json(resp, first=True)
                return self._coerce_types(parsed, self.state_schema)
            except Exception:
                if attempt == retries - 1:
                    raise


    def parse_action(self, text: str) -> Dict[str, Any]:
        """
        解析 Agent 的输出为结构化 action。

        Example (regex parser):
        >>> parse_action("I choose 37")
        {"choice": 37}

        Example (json parser):
        >>> parse_action('{"choice": 42}')
        {"choice": 42}

        Example text 直出
        >>> parse_action("I choose 37")
        {"choice": "I choose 37"}
        """
        parser_type = self.action_parser_cfg.get("type")

        if parser_type == "regex":
            pattern = self.action_parser_cfg["pattern"]
            match = re.search(pattern, text)
            if not match:
                raise ValueError(f"Regex parse failed: {text}")
            value = match.group(1)
            return self._values_to_action_dict([value])

        elif parser_type == "json":
            parsed = extract_json(text, first=True)
            return self._coerce_types(parsed, self.action_schema)

        elif parser_type == "split":
            sep = self.action_parser_cfg.get("sep", ",")
            parts = text.strip().split(sep)
            return self._values_to_action_dict(parts)
        
        elif parser_type == "text":
            # 直接把原始文本放进 schema 的第一个字段
            return self._values_to_action_dict([text.strip()])

        else:
            raise NotImplementedError(f"Unsupported action parser: {parser_type}")


    def _values_to_action_dict(self, values: List[Any]) -> Dict[str, Any]:
        """
        将解析出来的值绑定到 action schema。

        Example:
        >>> _values_to_action_dict(["37"])
        {"choice": 37}
        """
        if len(values) != len(self.action_schema):
            raise ValueError("Value count does not match action schema")

        result = {}
        for f, v in zip(self.action_schema, values):
            result[f["name"]] = self._coerce_by_type(v, f["type"])
        return result


    def _coerce_types(self, parsed: Dict[str, Any], schema: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        按 schema 对解析结果做类型转换。

        Example:
        >>> parsed = {"round": "2", "target": "35.2"}
        >>> schema = [{"name":"round","type":"int"},{"name":"target","type":"float"}]
        >>> _coerce_types(parsed, schema)
        {"round": 2, "target": 35.2}
        """
        result = {}
        for f in schema:
            name, typ = f["name"], f["type"]
            if name in parsed:
                result[name] = self._coerce_by_type(parsed[name], typ)
        return result


    def _coerce_by_type(self, value: Any, typ: str) -> Any:
        """
        类型转换。

        Example:
        >>> _coerce_by_type("42", "int")
        42
        >>> _coerce_by_type(["1","2"], "list[int]")
        [1,2]
        """
        if typ == "int":
            return int(value)
        elif typ == "float":
            return float(value)
        elif typ == "str":
            return str(value)
        elif typ == "bool":
            return bool(value)
        elif typ.startswith("list["):
            subtype = typ[5:-1]
            return [self._coerce_by_type(v, subtype) for v in value]
        else:
            return value
