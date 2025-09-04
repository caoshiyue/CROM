##
# Author:  
# Description:  
# LastEditors: Shiyuec
# LastEditTime: 2025-09-02 06:39:15
## 
import re
import json

def extract_json(text: str, first: bool = True) -> dict:
    """
    从 LLM 输出中提取 JSON。
    
    - 支持 markdown ```json ... ``` 格式
    - 支持前后杂质
    - 支持多段 JSON，默认取第一个
    """
    md_match = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if md_match:
        text = md_match[0] if first else md_match[-1]

    def find_all_json_objects(s: str) -> list:
        json_objects = []
        brace_level = 0
        start_index = -1
        
        for i, char in enumerate(s):
            if char == '{':
                if brace_level == 0:
                    start_index = i
                brace_level += 1
            elif char == '}':
                brace_level -= 1
                if brace_level == 0 and start_index != -1:
                    json_str = s[start_index : i + 1]
                    try:
                        parsed_json = json.loads(json_str)
                        json_objects.append(parsed_json)
                    except json.JSONDecodeError:
                        pass
                    start_index = -1
        return json_objects

    all_jsons = find_all_json_objects(text)
    
    if not all_jsons:
        raise ValueError("No valid JSON object found in text")

    return all_jsons[0] if first else all_jsons[-1]
