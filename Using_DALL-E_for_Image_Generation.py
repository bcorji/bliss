import openai

# Set up your OpenAI API key
openai.api_key = "your-openai-api-key"

# Generate image with DALL-E
response = openai.Image.create(
    prompt="A futuristic cityscape with flying cars",
    n=1,
    size="1024x1024"
)

# Print the URL of the generated image
image_url = response['data'][0]['url']
print(image_url)