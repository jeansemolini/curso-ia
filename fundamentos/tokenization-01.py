import nltk

nltk.download("punkt_tab")

text = "Machine learning é um campo da inteligência artificial que permite que computadores aprendam padrões a partir de dados. Sem serem programados explicitamente para cada tarefa."

word_tokens = nltk.word_tokenize(text, language="portuguese")
print(word_tokens)

sentence_tokens = nltk.sent_tokenize(text, language="portuguese")
print(sentence_tokens)


def preprocess(text):
    tokens = nltk.word_tokenize(text.lower(), language="portuguese")
    return [word for word in tokens if word.isalnum()]


documents = [
    "Machine learning é o aprendizado automático de máquinas a partir de dados.",
    "Ele permite que sistemas façam previsões e decisões sem programação explícita.",
    "É usado em áreas como reconhecimento de voz, imagens e recomendação de conteúdo.",
]

processed_docs = [" ".join(preprocess(doc)) for doc in documents]
print(processed_docs)
