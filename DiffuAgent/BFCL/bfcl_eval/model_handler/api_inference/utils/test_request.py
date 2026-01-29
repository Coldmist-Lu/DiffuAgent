from request_llm import REQUEST_LLM
from request_dllm import REQUEST_DLLM

# llm = REQUEST_LLM(model_path="/model/ModelScope/Qwen/Qwen3-8B",
# llm = REQUEST_LLM(model_path="/model/ModelScope/Qwen/Qwen3-8B",
llm = REQUEST_DLLM(model_name="Dream")

messages = [
    {
        "role": "system",
        "content": (
            "You are an expert in composing functions. "
            "You are given a question and a set of possible functions. "
            "Based on the question, you will need to make one or more function/tool calls to achieve the purpose.\n"
            "If none of the functions can be used, point it out. "
            "If the given question lacks the parameters required by the function, also point it out.\n"
            "You should only return the function calls in your response.\n\n"
            "If you decide to invoke any of the function(s), you MUST put it in the format of "
            "[func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]\n"
            "You SHOULD NOT include any other text in the response."
        )
    },
    {
        "role": "user",
        "content": "Find the area of a triangle with a base of 10 units and height of 5 units."
    }
]

tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate_triangle_area",
            "description": "Calculate the area of a triangle given its base and height.",
            "parameters": {
                "type": "object",
                "properties": {
                    "base": {
                        "type": "integer",
                        "description": "The base of the triangle."
                    },
                    "height": {
                        "type": "integer",
                        "description": "The height of the triangle."
                    },
                    "unit": {
                        "type": "string",
                        "description": "The unit of measure (defaults to 'units' if not specified)."
                    }
                },
                "required": ["base", "height"]
            }
        }
    }
]

message = [{'role': 'system', 'content': "You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.\nIf none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.\nYou should only return the function calls in your response.\n\nIf you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]\nYou SHOULD NOT include any other text in the response.\n\nAt each turn, you should try your best to complete the tasks requested by the user within the current turn. Continue to output functions to call until you have fulfilled the user's request to the best of your ability. Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task.\n\nHere is a list of functions in JSON format that you can invoke.\nYou must NOT include <think> or any reasoning. Your response must ONLY contain function calls in the required format."}, {'role': 'user', 'content': 'Find me a Rock concert in Chicago with ticket availability under $100.'}]
tools = [{'name': 'find_concert', 'description': 'Locate a concert in a specified location within a certain budget. Note that the provided function is in Python 3 syntax.', 'parameters': {'type': 'dict', 'properties': {'location': {'type': 'string', 'description': 'The city where you are looking for a concert. In the format City, State.'}, 'price': {'type': 'integer', 'description': 'Maximum ticket price.'}, 'genre': {'type': 'string', 'description': "Music genre of the concert. Default to 'Jazz'. ", 'enum': ['Rock', 'Pop', 'Country', 'Jazz', 'Classical']}}, 'required': ['location', 'price']}}]

messages = [{'role': 'system', 'content': "You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.\nIf none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.\nYou should only return the function calls in your response.\n\nIf you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]\nYou SHOULD NOT include any other text in the response.\n\nAt each turn, you should try your best to complete the tasks requested by the user within the current turn. Continue to output functions to call until you have fulfilled the user's request to the best of your ability. Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task.\n\nHere is a list of functions in JSON format that you can invoke.\nYou must NOT include <think> or any reasoning. Your response must ONLY contain function calls in the required format."}, {'role': 'user', 'content': 'Calculate the factorial of 5 using math functions.'}]

tools = [{'type': 'function', 'function': {'name': 'math_factorial', 'description': 'Calculate the factorial of a given number. Note that the provided function is in Python 3 syntax.', 'parameters': {'type': 'object', 'properties': {'number': {'type': 'integer', 'description': 'The number for which factorial needs to be calculated.'}}, 'required': ['number']}}}]

# response = llm.chat_completion_with_tools(messages = messages, tools = tools)
# response = llm.num_tokens_from_messages_with_tools(messages = messages, tools = tools)

# print(response)
# response = llm.num_tokens_from_messages(messages = messages)

# print(response)

message = [{"role": "user", "content": "A store sells apples for $2 each and oranges for $3 each. Tom bought 5 apples and 4 oranges. How much did he spend in total? Please solve this step by step."}]

message[0]["content"] *= int(16384/40)

response = llm.num_tokens_from_messages(messages = message)
print(response)

response = llm.chat_completion(message)
print(response.text)
