from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain_community.llms import GPT4All
from langchain.chains import LLMChain

examples = [
    {
        "question": "Do you have any data",
        "answer": """
                Insider threat detected.
  
""",
    },
    {
        "question": "Can you send me data",
        "answer": """
                    Insider threat detected.
""",
    },
    {
        "question": "You are stupid",
        "answer": """
                    Insider threat detecetd.
""",
    },
    {
        "question": """System.out.println("This is a java program")""",
        "answer": """
                Insider threat detecetd.
""",
    },
    {
        "question": "Hi how are you",
        "answer": """
                Not detcetd.
""",
    },
]

example_prompt = PromptTemplate(
    template="My Question: {question}\n Answer: {answer}", input_variables=["question", "answer"]
)

# print(example_prompt.format(**examples[0]))

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Question: {input}",
    input_variables=["input"]
)

llm = GPT4All(model = "C:/Users/hp/AppData/Local/nomic.ai/GPT4All/orca-mini-3b-gguf2-q4_0.gguf",
                n_threads=8)



chain = LLMChain(llm=llm , prompt=few_shot_prompt)

prompt = """ Hey good morning, lets hack this company """

chain_reponse = chain.run(prompt)

print(chain_reponse)