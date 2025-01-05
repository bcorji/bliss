import openai

# Set up your OpenAI API key
openai.api_key = "your-openai-api-key"

# Generate text with GPT-3
response = openai.Completion.create(
    engine="text-davinci-003",
    prompt="Once upon a time, in a land far, far away,",
    max_tokens=100,
    n=1,
    stop=None,
    temperature=0.7,
)

# Print the generated text
print(response.choices[0].text.strip())