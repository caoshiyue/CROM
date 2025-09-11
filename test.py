总之，不管接口如何，区别只有2点，1、参数不同，2. response 不同；在我的代码中一个完整的请求被分成3部分：
    async def chat(self, model: str, messages: List[Dict[str, str]], **kwargs) -> str:
        """Main method to get a chat completion. (REFACTORED)"""
        #预处理
        provider, model_name, mode, request_kwargs = self._parse_and_prepare_request(model, messages=messages, **kwargs) #   model_name,messages 已经吸收进request_kwargs
        client = self._get_client(provider)
        #API请求
        completion = await self._perform_api_call(client, 'chat', **request_kwargs)
        if mode == "DEBUG":
            self._handle_debug_log(model_name, provider, messages, completion)
        #后处理
        is_stream = request_kwargs.get('stream', False)
        return await self._process_response(provider, completion, is_stream)

那么，保持请求部分不变， 你只需要帮我写 预处理（即根据输入参数分辨是completion 还是 response，函数返回增加“request_type”变量），后处理（根据request_type 处理api的返回，无论是completion还是response）
