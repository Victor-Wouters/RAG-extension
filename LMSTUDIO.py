import os
import openai

def augment_response(query, document):


  if True:
    prompt= "Only use the following information to provide an answer on the final question in maximum 2 sentences, if no information provided, don't give an answer to the question, do not use your own knowledge: "

  openai.api_base = "http://localhost:1234/v1" # point to the local server
  openai.api_key = "" # no need for an API key

  completion = openai.ChatCompletion.create(
    model="local-model", 
    messages=[
      {"role": "system", "content": 'Follow the instructions very very careful, always answer in less than 50 words, only use the provided information, never use your own knowledge.'},
      {"role": "user", "content": str(prompt) + ' ' + str(document) + ' ' + str(query)}
    ]
  )
  return completion['choices'][0]['message']['content']