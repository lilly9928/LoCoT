
def Prompt4LM(model,num_object,num_llm_p,cot,datatype=""): 
    
    if 'vicuna' in model:
        start_word = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER:"
        end_word = 'ASSISTANT:'

    elif model =="llava-hf/llava-v1.6-vicuna-7b-hf" :
        start_word = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER:"
        end_word = 'ASSISTANT:'
        
    elif model =="llava-hf/llava-1.5-7b-hf" :
        #"<|im_start|>system\n<your_system_prompt_here><|im_end|><|im_start|>user\n<image>\n<your_text_prompt_here><|im_end|><|im_start|>assistant\n"
        start_word = "<|im_start|>user"
        end_word = '<|im_end|><|im_start|>assistant\n'
    
    elif model =="llava-hf/llama3-llava-next-8b-hf" :
        start_word = "[INST]"
        end_word = '[/INST]'
     
    else:
        start_word = "[INST]"
        end_word = '[/INST]'
     

    
    if cot =='no':
        PROMPT_CONCLUSION_GENERATION = (
            f"{start_word}<image>\n"
            "Answer the question using a single word or phrase.\n"
            "Question: \"{question}\"\n"
            f"{end_word}"
        )

    elif cot =='cot':
        PROMPT_CONCLUSION_GENERATION = (
            f"{start_word}<image>\n"
            "Answer the question using a single word or phrase.\n"
            "Question: \"{question}\"\n"
            "Let's think step by step."
            f"{end_word}"
        )

    elif cot =='long':
        PROMPT_CONCLUSION_GENERATION = (
            f"{start_word}<image>\n"
            "Respond to the inquiry presented by providing your answer in a succinct and focused format, specifically limiting your response to either a single word or a short phrase. Ensure clarity and directness in your reply, avoiding any extraneous information or elaboration beyond the precise answer required by the question posed.\n"
            "Question: \"{question}\"\n"
            f"{end_word}"
        )
    

    PROMPT_OBJECT_SELECTION =  (
        f"{start_word} <image>\n" 
            "Select object from {object} which is related to question '{question}'."
        "Select one word provide in {object}."
            f"{end_word}"
    )
    PROMPT_VLM_PREMISES = prepare_object_premises_prompt(num_premises=num_object,instruction=start_word,endingword=end_word)
    PROMPT_LLM_PREMISES = prepare_llm_premises_prompt(num_llm_p)
    PROMPT_SCORING = (
        f"{start_word} <image>\n"
        "Given the following:\n\n"
        "- Information: \"{information}\"\n\n"
        "Choose the most appropriate score to indicate whether the provided information is reasonable to image. Consider the image as well. Return only the letter corresponding to your choice from the options below:\n\n"
        "- (a) 0.1\n"
        "- (b) 0.5\n"
        "- (c) 0.7\n"
        "- (d) 0.9\n"
        "- (e) 0.95\n"
        "- (f) unknown\n\n"
        "Example choice format: (a)\n"
        f"{end_word}"
    )
    PROMPT_INFERENCE = (
        f"{start_word} <image>\n"
        "Answer the question using a single word or phrase.\n"
        "Information: \" {premise}\n\n"
        "Question: \"{question}\"\n"
        f"{end_word}"
    )

    if datatype == 'choice':

        PROMPT_INFERENCE = (
        f"{start_word}<image>\n"
        "Question: \"{question}\"\n"
        "Information: \" {premise}\n\n"
        "{options}"
        "Answer with the option's letter from the given choices directly.\n"
        f"{end_word}"
    )




    return PROMPT_CONCLUSION_GENERATION, PROMPT_OBJECT_SELECTION, PROMPT_VLM_PREMISES, PROMPT_LLM_PREMISES,PROMPT_SCORING, PROMPT_INFERENCE






VLM_PROMPT_FORMAT_ANSWER_onlyVLM = (
    "[INST] <image> Answer the following question based on the image. "
    "The answer must be concise, without any explanation, and using a maximum of 3 to 5 words. "
    "Question: \"{question}\"[/INST]"
)

VLM_PROMPT_FORMAT_OBJECT_SELECT = (
    "[INST] <image> Select object from {object} which is related to question '{question}'."
    "Select one word provide in {object}."
    "[/INST]"
)

VLM_PROMPT_FORMAT_OBJECT_SELECT_vlm = (
    "[INST] <image> Choose one object from image, which is related to question '{question}'. "
    "Format is <object>"
    "[/INST]"
)


LLM_PROMPT_RULE_GENERATION = (
    "Provide 3 short explanations about the conditions and properties of the answer and question which can be known in image. "
    "Each explanation must be concise, simple, and relevant to the image. "
    "Use a maximum of 10 words per explanation. "
    "Avoid detailed descriptions and keep the explanations straightforward. "
    "Answer in this format: 1. explanation1 2. explanation2 3. explanation3\n\n"
    "Question: \"{question}\"\n"
    "Answer: \"{answer}\"\n\n"
)

VLM_PROMPT_FORMAT_SCORE = (
    "[INST] <image>\n"
    "Given the following:\n\n"
    "- Information: \"{information}\"\n\n"
    "Choose the most appropriate score to indicate whether the provided information is reasonable to image. Consider the image as well. Return only the letter corresponding to your choice from the options below:\n\n"
    "- (a) 0.1\n"
    "- (b) 0.5\n"
    "- (c) 0.7\n"
    "- (d) 0.9\n"
    "- (e) 0.95\n"
    "- (f) unknown\n\n"
    "Example choice format: (a)\n"
    "[/INST]"
)

LLM_PROMPT_FORMAT_SCORE = (
    "Given the following:\n\n"
    "Information: \"{information}\"\n\n"
    "Choose the most appropriate score to indicate whether the provided information is fact to {conclusion}.Return only the letter corresponding to your choice from the options below:\n\n"
    "- (a) 0.1\n"
    "- (b) 0.3\n"
    "- (c) 0.85\n"
    "- (d) 0.9\n"
    "- (e) 0.95\n"
    "- (f) unknown\n\n"
    "Example choice format: (a)\n"
)


LLAVA_PROMPT_FORMAT_ANSWER_PREMISE = (
  "[INST] <image>\n"
    "Provide 3 brief reason for the answer using visual commonsence. "
    "Each explanation must be concise, simple, and relevant to the {object}. "
    "Use a maximum of 10 words per explanation. "
    "Avoid detailed descriptions and keep the explanations straightforward. "
    "Answer in this format: 1. explanation1 2. explanation2 3. explanation3 \n\n"
    "Question: \"{question}\"\n"
    "Answer: \"{answer}\"\n\n"
    "This is because"
    "[/INST]"
)

VLM_PROMPT_FORMAT_FINAL_ANSWER = (
    "[INST] <image>\n"
    "Answer the question using a single word or phrase.\n"
    "Information: \" {premise}\n\n"
    "Question: \"{question}\"\n"
    "[/INST]"

)

VLM_PROMPT_FORMAT_FINAL_ANSWER_onlyvlm = (
    "[INST] <image>\n"
     "Answer the question using a single word or phrase.\n"
     "Question: \"{question}\"\n"
    "[/INST]"
)

VLM_PROMPT_FORMAT_FINAL_ANSWER_onlyvlm_cot = (
    "[INST] <image>\n"
     "Answer the question using a single word or phrase.\n"
     "Question: \"{question}\"\n"
     "Let's think step by step."
    "[/INST]"
)

# 윤오빠 
VLM_PROMPT_FORMAT_FINAL_ANSWER_onlyvlm_long_cot = (
    "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.USER: <image>\n"
     #"Respond to the inquiry presented by providing your answer in a succinct and focused format, specifically limiting your response to either a single word or a short phrase. Ensure clarity and directness in your reply, avoiding any extraneous information or elaboration beyond the precise answer required by the question posed.\n"
     "Answer the question using a single word or phrase.\n"
     "\"{question}\"\n"
    "ASSISTANT:"
)


LLM_PROMPT_FORMAT_ANSWER = (
    "Answer the question using a single word or phrase.\n"
    "Information: \" {caption}\n\n"
    "Question: \"{question}\"\n"
    "Answer: "
)

LLM_PROMPT_FORMAT_FINAL_ANSWER = (
    "Answer the question using a single word or phrase.\n"
    "Context: \" {caption}\n\n"
    "Knowledge: \" {premise}\n\n"
    "Question: \"{question}\"\n"
    "Answer: "
)



LLM_PROMPT_FORMAT_PREMISE_ver1 = (
    "From the image, relevance object is {object}.\n"
    "When '{answer}' is the answer to '{question}', answer 5 visual facts that can be confirmed from the image. \n\n"
    "Answer in one sentence."
)

LLM_PROMPT_FORMAT_PREMISE_ver2 = (
    "From the image, relevance object is {object}.\n"
    "When '{answer}' is the answer to '{question}'. What could be the relevance visual facts. Answer with 5 concise, highly related phrases. \n\n"
    "The format is '<object> <verb> <attribute/position/etc>'."
)

LLM_PROMPT_FORMAT_PREMISE_ver3 = (
    "From the image, relevance object is {object}.\n"
    "When '{answer}' is the answer to '{question}'. What could be the relevance visual premises. Answer with 5 concise, highly related phrases. \n\n"
    "The format is '<object> <verb> <attribute>'."
)

def prepare_object_premises_prompt(num_premises,instruction='[/INST]',endingword='[/INST]'):
    explanation = ''
    for i in range(1,num_premises+1):
        explanation += f'{i}. explanation{i} '

    prepared_prompt = (
        f"{instruction} <image>\n"
        f"Provide {num_premises} brief reason for the answer using visual commonsence. \n"
        "Each explanation must be concise, simple, and relevant to the {object}. \n"
        "Use a maximum of 10 words per explanation. \n"
        "Avoid detailed descriptions and keep the explanations straightforward. \n"
        f"Answer in this format: {explanation.strip()}.\n\n"
        "Question: \"{question}\"\n"
        "Answer: \"{answer}\"\n\n"
        f"{endingword} This is because \n"
        
        )
    
    return prepared_prompt



def prepare_llm_premises_prompt(num_premises):
    
    explanation = ''
    # examples = ['The park is spacious']
    # if num_premises > 1:
    for i in range(1,num_premises+1):
        explanation += f'{i}. explanation{i} '
    explanation = explanation.strip()
    
    examples = [
        "1. The park is spacious\n",
        "2. Suitable for outdoor activities\n",
        "3. Safe environment for cycling\n",
        "4. Ideal for a picnic\n",
        "5. Enjoy nature walks\n",
    ]
        
    LLM_PROMPT_RULE_GENERATION = (
        f"Provide {num_premises} short explanations about the conditions and properties of the answer and question which can be known in image. "
        "Each explanation must be concise, simple, and relevant to the image. "
        "Use a maximum of 10 words per explanation. "
        "Avoid detailed descriptions and keep the explanations straightforward. "
        f"Answer in this format: {explanation}.\n\n"
        "Example:\n"
        "-  Question: \"What activities are suitable for this place?\"\n"
        "-  Answer: \"Cycling\"\n\n"
        f"{''.join(examples[:num_premises])}\n"
        "Your turn:\n"
        "-  Question: \"{question}\"\n"
        "-  Answer: \"{answer}\"\n\n"
    )
    
    return LLM_PROMPT_RULE_GENERATION