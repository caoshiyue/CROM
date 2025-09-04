from response import *

msg= """
"You are Alex and involved in a survive challenge.You are one of five players in the game. Everyone is required to choose an integer between 1 and 100 in each round. The player whose chosen number is closest to (0.8 * the average of all chosen numbers) wins the round without any HP deduction. All other players will have 1 HP deducted. However, if all players choose the same number, their health points are deducted together."
"The following is previous results of game: In Round 2, you chose 26, Bob chose 23, Cindy chose 28, David chose 23, and Eric chose 27, The average is (26 + 23 + 28 + 23 + 27) / 5 = 25.4. resulting in Bob and David winning the round while you and Eric lost 1 HP each.\n Now You are Alex, one of five players in the survive challenge, currently with 8 HP. Bob, Cindy, David, and Eric have 9, 9, 9, and 8 HP respectively.\n "
"You need to choose an integer between 1 and 100 for Round 3. Let's think step by step, and finally make a decision."
"""

prompt= [{'role': 'user', 'content': msg}]

response = openai_response(
    model="xiaoai:gpt-4o-mini" , #meta-llama/llama-3.1-8b-instruct
    messages=prompt,
    max_tokens=1600,
    temperature=0.7,
    top_p=0.9,
    frequency_penalty=0,
    presence_penalty=0,
    stop=None
)
print(response)