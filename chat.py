from response import *

msg= """
帮我查一下目前特斯拉(Tesla)的股价
"""

prompt= [{'role': 'user', 'content': msg}]

response = openai_response(
    model="openrouter:openai/gpt-4o-mini-search-preview" , #meta-llama/llama-3.1-8b-instruct
    messages=prompt,
    web_search_options={  # 关键参数：启用搜索
        "search_context_size": "medium"   # 可选: "low" / "medium" / "high"
    },
    max_tokens=1600,
    temperature=0.7,
    top_p=0.9,
    frequency_penalty=0,
    presence_penalty=0,
    stop=None
)
print(response)