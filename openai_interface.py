import openai
from openai.error import RateLimitError
import logging
import os
from tqdm import tqdm

'''
    An interface to OpenAI models to predict text and provide an interactive chat interface.
    Supported models: gpt3-davinci (davinci), ChatGPT (chat), GPT-4 (chat-gpt4)

    Author: Stephen Moskal (smoskal@csail.mit.edu)
    MIT CSAIL ALFA Group
'''


class OpenAIInterface:

    def __init__(self, api_key=None):
        if api_key:
            openai.api_key = api_key
        else:
            openai.api_key = os.getenv("OPENAI_API_KEY")

        if not openai.api_key:
            logging.getLogger('sys').warning(f'[WARN] OpenAIInterface (Init): No OpenAI API key given!')

    def predict_text(self, prompt, max_tokens=100, temp=0.8, mode='chat', prompt_as_chat=False):
        '''
        Queries OpenAI's GPT-3 model given the prompt and returns the prediction.
        See: openai.Completion.create()
            engine="text-davinci-002"
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0

        @param prompt: Prompt describing what you would like out of the
        @param max_tokens: Length of the response
        @return:
        '''

        try:
            if mode == 'chat':
                if prompt_as_chat:
                    message = prompt
                else:
                    message = [{"role": "user", "content": prompt}]

                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=message,
                    temperature=temp
                )
                return response['choices'][0]['message']['content']
            elif mode == 'chat-gpt4':
                if prompt_as_chat:
                    message = prompt
                else:
                    message = [{"role": "user", "content": prompt}]

                response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=message,
                temperature=temp
                )
                return response['choices'][0]['message']['content']
            else:
                response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=prompt,
                    temperature=temp,
                    max_tokens=max_tokens,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )

                return response.choices[0].text
        except RateLimitError as e:
            logging.getLogger('sys').error(f'[WARN] OpenAIInterface: Rate Limit Exceeded! {e}')
            return '0'
        except Exception as e:
            logging.getLogger('sys').error(f'[ERROR] OpenAIInterface: Unexpected exception- {e}')
            return '-1'

    def prompt_chain(self, prompts: list, max_tokens=300, temp=.8, mode='chat', system_prompt=''):
        '''
        Takes in a list of prompts which are executed in order and maintains the chat history
        with each sequential prompt.  This allows us to perform chain of thought type prompts
        more seamlessly.
        :param prompts: list of prompts that are run sequentially
        :param max_tokens:
        :param temp:
        :param mode: chat, chat-gpt4
        :param system_prompt: the system prompt (optional)
        :return:
        '''
        prompt_history = [{'role': 'system', 'content': system_prompt}]
        response_history = []
        for sub_prompt in tqdm(prompts):
            prompt_format = {'role': 'user', 'content': sub_prompt}
            prompt_history.append(prompt_format)
            model_response = self.predict_text(prompt_history, max_tokens=max_tokens, temp=temp, mode=mode, prompt_as_chat=True)
            response_history.append(model_response)
            response_format = {'role': 'assistant', 'content': model_response}
            prompt_history.append(response_format)
        return response_history, prompt_history

    def start_interactive_chat(self, chat_model='chat', max_response_tokens=500, quiet_mode=False):
        chat_history = []

        print("Input your system prompt (q to quit):")
        system_prompt_text = input()
        if system_prompt_text == ('q' or 'quit'):
            print('Aborting chat mode...')
            return
        system_prompt = {'role': 'system', 'content': system_prompt_text}
        chat_history.append(system_prompt)

        while True:

            if not quiet_mode:
                print('Prompt the model (q to quit, h for history, r to reset history):')
            prompt_text = input()

            if prompt_text == ('q' or 'quit'):
                print('Aborting chat mode...')
                return
            elif prompt_text == ('h' or 'history'):
                print(chat_history)
            elif prompt_text == ('r' or 'reset'):
                print('Resetting the chat history')
                chat_history = [system_prompt]
            else:
                try:
                    prompt_format = {'role': 'user', 'content': prompt_text}
                    chat_history.append(prompt_format)
                    chat_response = self.predict_text(chat_history, max_tokens=max_response_tokens, mode=chat_model,
                                                      prompt_as_chat=True)
                    print(f'{chat_response}\n')
                    chat_history.append({'role': 'assistant', 'content': chat_response})
                except Exception as e:
                    print(f'Unknown exception: {e}')
                    print('Do you want to continue? (y/n)')
                    selection = input()
                    if selection == 'n':
                        print('Aborting chat mode...')
                        return

    def get_embedding(self, input_text, model="text-embedding-ada-002"):
        temp_text = input_text.replace("\n", " ")
        return openai.Embedding.create(input=[temp_text], model=model)['data'][0]['embedding']

if __name__ == '__main__' :

    personal_api_key = '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'
    openai_interface = OpenAIInterface(api_key=personal_api_key)