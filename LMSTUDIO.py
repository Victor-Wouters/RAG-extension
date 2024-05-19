import os
import openai

def augment_response(query, document):

  # Best prompt for the RAG task
  if True:
    prompt= "Only use the following information to provide an answer on the final question in maximum 2 sentences, if no information provided, don't give an answer to the question, do not use your own knowledge: "
  # Test prompt
  if False:
    prompt = "Only answer to the question with information provided in the prompt. "
  # Test without prompt
  if False: 
    prompt= ""
  # Test without context
  if False:
    document = " "   
  # Test with unrelevant document
  if False:
    document = "NAME: basic pancake mix INGREDIENTS: all-purpose flour, sugar, baking powder, baking soda, salt, milk, eggs, vanilla, vegetable oil STEPS: in a bowl , mix together all the dry ingredients, make a well in the centre and pour in the milk, start with 1 1 / 4 cups milk , adding up to another 1 / 4 cup if necessary , as you mix it with the flour, add the two eggs , vanilla if using and oil , whisking until mixed but still a bit lumpy, heat a frying pan and when hot , pour in some pancake mix, how much depends on how experienced you are at flipping pancakes and how big you want them, we do about 1 / 4 cup a time for small , easy-to-flip pancakes but you could make this as much as 1 / 2 cup of pancake mix, if you are adding fruit , i like to sprinkle it on top of the pancake now, when the pancake starts to bubble on top and is golden brown on the cooked side , turn it and continue cooking until both sides are golden brown, the first pancake is always a bit of a test so adjust the batter by adding more flour if you need to make it thicker or more milk if you want a thinner pancake, in either case , just add a few spoonfuls at a time until you get it right, repeat the cooking process with the remaining batter, you may need to adjust the heat as pan tends to get hotter as you keep making pancakes, keep the cooked pancakes covered with a tea towel , to keep them warm while you finish cooking the rest TAGS: 30-minutes-or-less, time-to-make, course, main-ingredient, cuisine, preparation, occasion, north-american, pancakes-and-waffles, breads, breakfast, eggs-dairy, easy, heirloom-historical, holiday-event, vegetarian, grains, eggs, stove-top, dietary, pasta-rice-and-grains, equipment DESCRIPTION: say goodbye to aunt jemima! once you taste these homemade pancakes, i think youll agree theyre much better than the boxed kind! add a bit of vanilla or fruit (frozen blueberries work well) to make them extra special. if you want, mix up the dry ingredients in advance and give it away as a gift (with instructions on how to finish making the pancakes) or just store it for even quicker pancakes on sunday mornings." # unrelated document

  openai.api_base = "http://localhost:1234/v1" # point to the local server
  openai.api_key = "" # no need for an API key

  completion = openai.ChatCompletion.create(
    model="local-model", 
    messages=[
      {"role": "system", "content": 'Follow the instructions very very careful, always answer in less than 50 words'},
      {"role": "user", "content": str(prompt) + ' ' + str(document) + ' ' + str(query)}
    ]
  )
  return completion['choices'][0]['message']['content']