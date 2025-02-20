# import os

# # import httpx
# from dotenv import load_dotenv
# from openai import AsyncAzureOpenAI, OpenAI
# from pydantic_ai.models.openai import OpenAIModel

# load_dotenv(override=True)

# # use_proxy = True

# # proxies = {
# #     'http://': os.getenv('PROXY_HTTP', 'http://mwg-hkidc.kucoin.net:9090'),
# #     'https://': os.getenv('PROXY_HTTPS', 'http://mwg-hkidc.kucoin.net:9090'),
# # }


# deepseek_client = OpenAI(
#     api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com"
# )

# gpt_client = AsyncAzureOpenAI(
#     azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
#     api_version=os.environ["AZURE_OPENAI_API_VERSION"],
#     api_key=os.environ["AZURE_OPENAI_API_KEY"],
#     # http_client=httpx.AsyncClient(proxies=proxies, verify=False) if use_proxy else None,
# )

# model_reason = OpenAIModel("deepseek-reasoner", openai_client=deepseek_client)

# model_4o = OpenAIModel("gpt4o", openai_client=gpt_client)
# model_0806 = OpenAIModel("gpt-4o-0806", openai_client=gpt_client)
# model_1120 = OpenAIModel("gpt-4o-1120", openai_client=gpt_client)
# model_mini = OpenAIModel("gpt-4o-mini", openai_client=gpt_client)


# if __name__ == "__main__":
#     from pydantic_ai import Agent

#     agent = Agent(model=model_1120)
#     res = agent.run_sync("who are you?")
#     print(res.data)
