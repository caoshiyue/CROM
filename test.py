##
# Author:  
# Description:  
# LastEditors: Shiyuec
# LastEditTime: 2025-09-02 02:26:44
## 
import json
import re

# 假设你的函数 extract_json 已经定义在这里

def test_extract_json():
    print("--- 开始测试 ---")

    # 1. 理想情况：纯净的 JSON 字符串
    print("\n--- 测试用例 1：理想情况（纯净 JSON） ---")
    text1 = '{"name": "Alice", "age": 30, "city": "New York"}'
    try:
        result1 = extract_json(text1)
        assert result1 == {"name": "Alice", "age": 30, "city": "New York"}
        print(f"✅ 通过！结果: {result1}")
    except Exception as e:
        print(f"❌ 失败！发生异常: {e}")


    # 2. 常见情况：包含 markdown 和前后杂质
    print("\n--- 测试用例 2：常见情况（带 markdown） ---")
    text2 = """
    这是前面的一些文本，AI 返回了以下 JSON：
    ```json
    {
      "product": "apple",
      "price": 2.5,
      "stock": 100
    }
    ```
    还有一些后续说明。
    """
    try:
        result2 = extract_json(text2)
        assert result2 == {"product": "apple", "price": 2.5, "stock": 100}
        print(f"✅ 通过！结果: {result2}")
    except Exception as e:
        print(f"❌ 失败！发生异常: {e}")
        
    # 3. 常见情况：不带 json 关键字的 markdown
    print("\n--- 测试用例 3：常见情况（不带 'json' 关键字的 markdown） ---")
    text3 = """
    这是另一个示例：
    ```
    {
      "status": "success",
      "data": [1, 2, 3]
    }
    ```
    """
    try:
        result3 = extract_json(text3)
        assert result3 == {"status": "success", "data": [1, 2, 3]}
        print(f"✅ 通过！结果: {result3}")
    except Exception as e:
        print(f"❌ 失败！发生异常: {e}")

    # 4. 常见情况：多个 JSON 对象，默认提取第一个
    print("\n--- 测试用例 4：常见情况（多 JSON，提取第一个） ---")
    text4 = """
    第一部分数据：{"id": 1, "value": "A"}
    第二部分数据：{"id": 2, "value": "B"}
    """
    try:
        result4 = extract_json(text4)
        assert result4 == {"id": 1, "value": "A"}
        print(f"✅ 通过！结果: {result4}")
    except Exception as e:
        print(f"❌ 失败！发生异常: {e}")
        
    # 5. 常见情况：提取最后一个 JSON 对象 (first=False)
    print("\n--- 测试用例 5：常见情况（多 JSON，提取最后一个） ---")
    try:
        result5 = extract_json(text4, first=False)
        assert result5 == {"id": 2, "value": "B"}
        print(f"✅ 通过！结果: {result5}")
    except Exception as e:
        print(f"❌ 失败！发生异常: {e}")

    # 6. 异常情况：JSON 格式错误
    print("\n--- 测试用例 6：异常情况（JSON 格式错误） ---")
    text6 = '{"key": "value", "invalid": "missing_quote}'
    try:
        extract_json(text6)
        print("❌ 失败！未捕获预期的 JSONDecodeError。")
    except ValueError as e:
        print(f"✅ 通过！成功捕获异常: {e}")

    # 7. 异常情况：无 JSON 对象
    print("\n--- 测试用例 7：异常情况（无 JSON） ---")
    text7 = "这是一段完全没有 JSON 的文本。"
    try:
        extract_json(text7)
        print("❌ 失败！未捕获预期的 ValueError。")
    except ValueError as e:
        print(f"✅ 通过！成功捕获异常: {e}")
        
    # 8. 额外测试：多行、复杂结构的 JSON
    print("\n--- 测试用例 8：复杂结构 JSON ---")
    text8 = """
    一些废话...
    ```json
    {
      "user_info": {
        "name": "Jane",
        "email": "jane@example.com"
      },
      "preferences": [
        {"type": "color", "value": "blue"},
        {"type": "theme", "value": "dark"}
      ]
    }
    ```
    """
    try:
        result8 = extract_json(text8)
        expected_result = {
            "user_info": {
                "name": "Jane",
                "email": "jane@example.com"
            },
            "preferences": [
                {"type": "color", "value": "blue"},
                {"type": "theme", "value": "dark"}
            ]
        }
        assert result8 == expected_result
        print(f"✅ 通过！结果: {result8}")
    except Exception as e:
        print(f"❌ 失败！发生异常: {e}")


    print("\n--- 所有测试完成 ---")

if __name__ == "__main__":
    def extract_json(text: str, first: bool = True) -> dict:
        """
        从 LLM 输出中提取 JSON。
        
        - 支持 markdown ```json ... ``` 格式
        - 支持前后杂质
        - 支持多段 JSON，默认取第一个
        """
        # 先处理 markdown 包裹 ```json ... ```
        md_match = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if md_match:
            text = md_match[0] if first else md_match[-1]

        # 使用计数器来匹配完整的 JSON 对象，支持嵌套
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
                        # 找到一个完整的 JSON 对象
                        json_str = s[start_index : i + 1]
                        try:
                            parsed_json = json.loads(json_str)
                            json_objects.append(parsed_json)
                        except json.JSONDecodeError:
                            # 如果解析失败，继续寻找下一个
                            pass
                        # 重置开始索引，以便寻找下一个独立的 JSON 对象
                        start_index = -1
            return json_objects

        all_jsons = find_all_json_objects(text)
        
        if not all_jsons:
            raise ValueError("No valid JSON object found in text")

        return all_jsons[0] if first else all_jsons[-1]

            
    test_extract_json()