import openai
import time
import os


class OpenAIModel(object):
    def __init__(self):
        with open('PREFS/api.key') as f:
            self.api_key = f.read().rstrip('\n')

    def generate(self, prompt, max_output_tokens):
        to_send = [{'role':'user', 'content':prompt}]
        client = openai.OpenAI(api_key=self.api_key)
        waittime = 2
        while True:
            try:
                response = client.chat.completions.create(
                  messages = to_send,
                  model='gpt-4-turbo-preview',
                  max_tokens=max_output_tokens,
                  temperature=0.7,
                  top_p=0.9,
                  )
                break
            except openai.RateLimitError as e:
                print(f'{e}: waiting {waittime}')
                time.sleep(waittime)
                waittime = min(waittime*2, waittime+30)
        output = response.choices[0].message.content
        return output
