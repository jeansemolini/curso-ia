import openai

client = openai.OpenAI(base_url="https://api.groq.com/openai/v1")

response = client.responses.create(
    model="llama-3.1-8b-instant",
    instructions="Responda de forma simples em apenas 1 paragrafo curto.",
    input="O que é machine learning?",
    temperature=0,
)

print(response.output)
print(response.output_text)
