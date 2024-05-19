import os
import openai

def summarize_chunk(document):

  # Best prompt for the RAG task
  if True:
    prompt= "Please provide a concise summary of the following academic paper in a hard maximum of 1 sentence. "
 
  openai.api_base = "http://localhost:1234/v1" # point to the local server
  openai.api_key = "" # no need for an API key

  completion = openai.ChatCompletion.create(
    model="local-model", 
    messages=[
      {"role": "system", "content": 'Please provide a concise summary of the following academic paper in maximum 2 sentences, this is a hard maximum.'},
      {"role": "user", "content": str(prompt) + ' ' + str(document)}
    ]
  )
  return completion['choices'][0]['message']['content']