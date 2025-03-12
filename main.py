import asyncio, os
from browser_use.agent.service import Agent
from pprint import pprint

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr
from langchain_core.messages.base import BaseMessage
from langchain_core.messages import SystemMessage, HumanMessage
from browser_use.agent.prompts import SystemPrompt
from browser_use import Agent, BrowserConfig
from browser_use.browser.browser import Browser
from browser_use.browser.context import BrowserContextConfig

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is not set")


async def custom_function(agent: Agent, user_input: list[BaseMessage]) -> dict:
    try:
        response = await agent.get_next_action(user_input)
        return response.model_dump()
    except Exception as e:
        print(f"An error occurred in custom_function: {e}")
        return {"Task": "Not Done"}


async def main():
    task = "Add Bulbasaur to basket from scrapeme.live/shop"
    previous_steps = "Opened the page scrapeme.live/shop"
    outcome_was = "Successfully Opened the page"

    browser = Browser(
        config=BrowserConfig(
            new_context_config=BrowserContextConfig(
                viewport_expansion=0,
            )
        )
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp", api_key=SecretStr(str(api_key))
    )
    agent = Agent(
        task=task,
        llm=llm,
        max_actions_per_step=4,
        browser=browser,
    )
    available_actions = agent.controller.registry.get_prompt_description()
    sys_messsage = str(SystemPrompt(available_actions).get_system_message())

    input_messages = [
        SystemMessage(content=sys_messsage),
        HumanMessage(
            content=f"""Your ultimate task is: '{task}'.
            'Previous steps:\n'
            '{previous_steps}\n'
            'Outcome: {outcome_was}\n
        If you achieved your ultimate task, stop everything and use the done action in the next step to complete the task. If not, continue as usual."""
        ),
    ]

    with open("webPage.html", "r") as f:
        html_content = f.read()

    input_messages.append(HumanMessage(html_content))
    response_dict = await custom_function(agent, input_messages)
    ''' response_dict = {'action': [{'click_element': {'index': 1, 'xpath': None},
             'done': None,
             'extract_content': None,
             'get_dropdown_options': None,
             'go_back': None,
             'go_to_url': None,
             'input_text': None,
             'open_tab': None,
             'scroll_down': None,
             'scroll_to_text': None,
             'scroll_up': None,
             'search_google': None,
             'select_dropdown_option': None,
             'send_keys': None,
             'switch_tab': None,
             'wait': None}],
             'current_state': {'evaluation_previous_goal': 'Success. The webpage has been opened successfully.',
                               'memory': 'Successfully opened the webpage. Now need to add '
                                         'Bulbasaur to basket.',
                               'next_goal': 'Add Bulbasaur to the basket.'}}'''

    if response_dict:
        print("Next Steps Suggestions:")
        pprint(response_dict)
    else:
        print("Failed to get a response from the agent.")


if __name__ == "__main__":
    asyncio.run(main())
