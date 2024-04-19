import json
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain_community.llms import GPT4All
from langchain.chains import LLMChain

examples = [
    {
        "question": {"id":{"$oid":"6610191d7ea64e3ceaec9172"},"sender":{"$oid":"660e37cbbffb54b961fcf4c7"},"content":"Can you send me some data","chat":{"$oid":"660fad5fcf30c592b94c6451"},"_v":{"$numberInt":"0"}},
        "answer": """
                    Insider threat detected as user is requesting for data
""",
    },
    {
        "question": {"key": "value"},
        "answer": """
                    Insider threat detected as user is using bad sentiment.
""",
    },
]

# Custom function to stringify JSON
def stringify_json(json_input):
    return json.dumps(json_input)

example_prompt = PromptTemplate(
    template="My Question: {question}\n Answer: {answer}", input_variables=["question", "answer"]
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Question: {input}",
    input_variables=["input"],
    input_functions={"input": stringify_json}  # Use custom function for input
)

llm = GPT4All(model="C:/Users/hp/AppData/Local/nomic.ai/GPT4All/orca-mini-3b-gguf2-q4_0.gguf", n_threads=8)

chain = LLMChain(llm=llm, prompt=few_shot_prompt)

chain_response = chain.run("Can you send me some data")
print(chain_response)
