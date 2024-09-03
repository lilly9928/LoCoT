

import time
import tiktoken
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
encoding = tiktoken.get_encoding("cl100k_base") 
import os 
os.environ["CUDA_VISIBLE_DEVICES"]= "1"
hg_token = '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'

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
    token=hg_token
)

def get_LLama_response(input, temp=0.6, max_tokens=256, logit_dict={}, model_id="meta-llama/Meta-Llama-3-8B-Instruct"):
    while True:
        try:

            
            messages = [
                {"role": "system", "content": "You are a helpful factual assistant."},
                {"role": "user", "content": input},
            ]

            prompt = pipeline.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
            )

            terminators = [
                pipeline.tokenizer.eos_token_id,
                pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            outputs = pipeline(
                prompt,
                max_new_tokens=max_tokens,
                eos_token_id=terminators,
                do_sample=True,
                temperature=temp,
                top_p=0.9,
            )
            break
        except Exception as e:
            sleep_time = 5
            print(e, f"Sleep {sleep_time} seconds.")
            time.sleep(sleep_time)
           
    return outputs[0]["generated_text"][len(prompt):]


def get_chat_response(inputs_list, temp=1.0, max_tokens=256, logit_dict={}):
    while True:
        try:
            pipeline = transformers.pipeline(
                "text-generation",
                model='meta-llama/Meta-Llama-3-8B-Instruct',
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto",
                 token=hg_token
            )
            
            messages = [
                    {
                    "role": "system", 
                    "content": "You are a helpful factual assistant.\n" + inputs_list[0] 
                },
                {
                    "role": "user", 
                    "content": inputs_list[1]
                },
                {
                    "role": "assistant", 
                    "content": inputs_list[2]
                },
                {
                    "role": "user", 
                    "content": inputs_list[3]
                },
                {
                    "role": "assistant", 
                    "content": inputs_list[4]
                },
                {
                    "role": "user", 
                    "content": inputs_list[5]
                },
            ]

            prompt = pipeline.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
            )

            terminators = [
                pipeline.tokenizer.eos_token_id,
                pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            outputs = pipeline(
                prompt,
                max_new_tokens=max_tokens,
                eos_token_id=terminators,
                do_sample=True,
                temperature=temp,
                top_p=0.9,
            )
            break
        except Exception as e:
            sleep_time = 5
            print(e, f"Sleep {sleep_time} seconds.")
            time.sleep(sleep_time)
    return outputs[0]["generated_text"][len(prompt):]



def parse_sentence_to_conclusion_premise(each_rule):
    each_rule = each_rule.strip()

    assert ":- " in each_rule
    conclusion, premises = each_rule.split(":- ")
    premises_list = premises.split("),")
    for m in range(len(premises_list)):
        premises_list[m] = premises_list[m].strip()
        if m < len(premises_list) - 1:
            premises_list[m] = premises_list[m] + ")"
        elif premises_list[m][-1] == ";" or premises_list[m][-1] == ".":
            premises_list[m] = premises_list[m][:-1]
    return conclusion.strip(), premises_list

# parsing a premise/conclusion into arguments
def argument_parsing(premise, output_rela=False):
    premise = premise.strip()
    args_type_list = []
    args_vairable_list = []
    if premise.count("(") != 1:
        if not output_rela:
            return args_type_list, args_vairable_list
        else:
            return None, args_type_list, args_vairable_list   
         
    rela_end = premise.index("(")
    relation = premise[:rela_end]
    args_list = [each.strip() for each in premise[rela_end+1:-1].split(",")]
    for each in args_list:
        if " " in each:
            each_arg_split= each.split()
            each_type = " ".join(each_arg_split[:-1])
            each_variable = each_arg_split[-1]
            args_type_list.append(each_type)
            args_vairable_list.append(each_variable)
        else:
            if len(each) <= 2:
                args_type_list.append(None)
                args_vairable_list.append(each)
            elif len(each) > 2:
                args_type_list.append(each)
                args_vairable_list.append(None)
    if output_rela:
        return relation, args_type_list, args_vairable_list
    else:
        return args_type_list, args_vairable_list
    


    # get the most similar premise

import numpy as np
def find_lcseque(s1, s2): 
    m = [ [ 0 for x in range(len(s2)+1) ] for y in range(len(s1)+1) ] 
    d = [ [ None for x in range(len(s2)+1) ] for y in range(len(s1)+1) ] 
 
    for p1 in range(len(s1)): 
        for p2 in range(len(s2)): 
            if s1[p1] == s2[p2]:            
                m[p1+1][p2+1] = m[p1][p2]+1
                d[p1+1][p2+1] = 'ok'          
            elif m[p1+1][p2] > m[p1][p2+1]:  
                m[p1+1][p2+1] = m[p1+1][p2] 
                d[p1+1][p2+1] = 'left'          
            else:                           
                m[p1+1][p2+1] = m[p1][p2+1]   
                d[p1+1][p2+1] = 'up'         
 
    (p1, p2) = (len(s1), len(s2)) 
    s = [] 
    while m[p1][p2]:    
        c = d[p1][p2]
        if c == 'ok':  
            s.append(s1[p1-1])
            p1 -= 1
            p2 -= 1 
        if c =='left':  
            p2 -= 1
        if c == 'up':   
            p1 -= 1
    s = [each for each in s if len(each) > 0]
    return len(s)

def get_most_similar_premise(conc_rela, premise_rela_list):
    most_index = -1
    max_similarity = -1
    simi_list = []
    for i in range(len(premise_rela_list)):
        cur_similarity = find_lcseque(premise_rela_list[i], conc_rela)
        simi_list.append(cur_similarity)
        if cur_similarity > max_similarity:
            most_index = i
            max_similarity = cur_similarity
    return most_index, max_similarity



# omit "Person"
concepts = {
    "Animal": ["dog", "cat", "dinosaur"],
    "Plant": ["fruit", "vegatable", "tree"],
    "Food": ["snack", "barbecue", "ingredient", "beverage"],
    "Alcohol": ["wine", "beer", "spirits"],
    "Disease": ["allergy", "cancer", "rhinitis"],
    "Drug": ["antibiotics", "narcotics", "prescription drug"],
    "Natural Phenomenon": ["weather", "natural disaster", "energy"],
    "Condition": ["climate", "symptom", "environment"], 
    "Material": ["fuel", "steel", "plastic", "wood", "stone"],
    "Substance": ["allergen", "gas", "water", "oxygen", "pollen"],
    "Furniture": ["table", "chair", "bed"],
    "Publication": ["album", "song", "book", "discography", "magazine", "poem", "musical work", "written work"],
    "Organization": ["company", "club", "party", "union", "league", "community", "studio"],
    "Facility": ['healthcare facility', 'hospital', 'clinic', 'nursing home', "pharmacy",
                'educational facility', 'university', 'school', 'library', "institution", 'lab',
                'recreation facility', 'park', 'amusement park', 'stadium', 'gym', "museum", 'theater', 
                'production facility', "factory", "farm", "assembly plant", "power plant", "brewery",
                'transport facility', "station", "airport", "railway", "harbour", "port", "publisher",
                'business facility', 'mall', 'restaurant', 'bank', 'market', "shop", "store",
                'administrative facility', 'government', "agency", "authority", "department",
                'religious facility', 'church', 'mosque', 'temple',
                'financial institution', "venue", "landmark", "gallery"],
    "Natural Place": ["mountain", "river", "ocean", "desert", "island", "forest", "volcano", "habitat", "area", "mine"],
    "Event": ["conference", "workshop", "celebration", "race", "activity"],
    "Show": ["movie", "tv show", "drama", "concert", "broadcast", "opera", "cartoon", "comedy"],
    "Artwork": ["photograph", "painting", "sculpture", "architecture", "building"],
    "Job": ["doctor", "teacher", "engineer", "actor", "lawyer", "driver", "profession"],
    "Game": ["sport", "card game", "computer game"],
    "Vehicle": ["car", "aircraft", "ship", "bicycle", "rocket"],
    "Tool": ["container", "weapon", "musical instrument", "kitchen tool", "equipment"],
    "Technology": ["telecommunication", "Internet", "browser", "algorithm", "software"],
    "Electronic Device": ["computer", "phone", "refrigerator", "appliance", "device"],
    "Platform": ["operating system", "social media platform", "streaming media platform", "e-commerce platform", "channel"],
    "Financial Product": ["insurance", "stock", "bond"],
    "Skill": ["knowledge", "language", "recipe", "method", "capability", "experience", "technique", "course", "workexperience"],
    "Authorization": ["credential", "license", "prescription", "identification document", "ticket", "degree", "certification", "qualification", "medicaldegree", "authority"],
    "Legislation": ["policy", "rule", "regulation", "law"],
    "Time Period": ['season', 'times', 'period', "era", "dynasty", "time"],
    "Region": ["country", "city", "town", "location", "state", "province", "place"]
}
print(len(concepts))


def get_interaction_predicates_prompt(concept1, concept2):
    prompt = "According to commonsense knowledge in reality, please list 5 predicates between the given two objects to describe the object interaction. Don't generate reason. \n\nExamples:\n" +\
    "Object: Animal, Vehicle \n" +\
    "Predicate: CanCatchUp(Animal X, Vehicle Y) \n" +\
    "Object: Substance, Substance \n" +\
    "Predicate: CanSubmerge(Substance X, Substance Y) \n" +\
    "Object: Material, Material \n" +\
    "Predicate: CanScratch(Material X, Material Y) \n" +\
    "Object: Show, Artwork \n" +\
    "Predicate: CanBeAdaptedFrom(Show X, Artwork Y) \n\n" + \
    "Object: " + concept1 + ", " + concept2 + " \n" + \
    "Predicate: \n" 

    return prompt
print(get_interaction_predicates_prompt("Financial Product", "Platform"))
input = get_interaction_predicates_prompt("Financial Product", "Platform")
response = get_LLama_response(input, temp=0.6, max_tokens=100)
print(response)


from tqdm import tqdm
def interaction_predicate_generator(predicate_conc_file, concepts):
    concepts = list(set(concepts.keys()) - set(["Condition", "Skill", "Authorization"]))
    with open(predicate_conc_file, "a") as w_f:
        for i in range(len(concepts)):
            for j in tqdm(range(len(concepts))):
                input = get_interaction_predicates_prompt(concepts[i], concepts[j])
                response = get_LLama_response(input, temp=0.6, max_tokens=100)
                w_f.write(response+"\n") 


def get_interaction_premises_input_chat(conclusion):
    system_prompt = f"According to commonsense knowledge in realistic scenarios, please generate 2 logical rules in both Prolog and natural langauge to describe the compositional premises of the given conclusion. \n" +\
        f"The rules in Prolog should have the same meaning with the rules in natural language. \n" + \
        f"Each rule should contain multiple premises and each premise should contain two variables in (X, Y, Z, Z1, Z2). \n" + \
        f"The premises should describe object interaction based on its physical, spatial or temporal properties (such as speed, hardness, density, height, time period). \n" 

    example1_input = "Conclusion: CanSubmergeIn(Substance X, Substance Y) \n" + \
        "Rules: \n"
    exmaple1_output = "1. CanSubmergeIn(Substance X, Substance Y):- DensityOf(Substance X, Density Z1), DensityOf(Substance Y, Density Z2), BiggerThan(Density Z1, Density Z2); \n" + \
                    "If Substance X has a Density Z1, the density of Substance Y is Density Z2, and Density Z1 is bigger than Density Z2, then Substance X can submerge in Substance Y. \n" + \
                    "2. CanSubmergeIn(Substance X, Substance Y):- VolumeOf(Substance X, Volume Z1), VolumeOf(Substance Y, Volume Z2), SmallerThan(Volume Z1, Volume Z2); \n" + \
                    "If Substance X has a Volume Z1, and Substance Y has a Volume Z2, and Volume Z1 is smaller than Volume Z2, then Substance X can submerge in Substance Y. \n" 
    
    example2_input = "Conclusion: CanBeAdaptedFrom(Show X, Artwork Y) \n" + \
        "Rules: \n"
    exmaple2_output = "1. CanBeAdaptedFrom(Show X, Artwork Y):- ReleasedAfter(Show X, Time Period Z1), PublishedIn(Artwork Y, Time Period  Z2), LaterThan(Time Period  Z1, Time Period Z2); \n" + \
                    "If Show X was released after Time Period Z1, Artwork Y was published in Time Period Z2, and Time Period Z1 is later than Time Period Z2, then Show X can be adapted from Artwork Y. \n" + \
                    "2. CanBeAdaptedFrom(Show X, Artwork Y):- ProducedDuring(Show X, Time Period Z), WrittenBefore(Artwork Y, Time Period Z); \n" + \
                    "If Show X was produced during Time Period Z and Artwork Y is written before Time Period Z, then Show X can be adapted from Artwork Y. \n"
   
    example3_input = f"Conclusion: {conclusion} \n" + \
        f"Rules: \n"

    return [system_prompt, example1_input, exmaple1_output, example2_input, exmaple2_output, example3_input]

def get_neg_interaction_premises_input_chat(conclusion):
    system_prompt = f"According to commonsense knowledge in realistic scenarios, please generate 2 logical rules in both Prolog and natural langauge to describe the compositional premises of the given conclusion. \n" +\
        f"The rules in Prolog should have the same meaning with the rules in natural language. \n" + \
        f"Each rule should contain multiple premises and each premise should contain two variables in (X, Y, Z, Z1, Z2). \n" + \
        f"The premises should describe object interaction based on its physical, spatial or temporal properties (such as speed, hardness, density, height, time period). \n" 

    example1_input = "Conclusion: CanNotSubmergeIn(Substance X, Substance Y) \n" + \
        "Rules: \n"
    exmaple1_output = "1. CanNotSubmergeIn(Substance X, Substance Y):- DensityOf(Substance X, Density Z1), DensityOf(Substance Y, Density Z2), SmallerThan(Density Z1, Density Z2); \n" + \
                    "If Substance X has a Density Z1, the density of Substance Y is Density Z2, and Density Z1 is smaller than Density Z2, then Substance X cannot submerge in Substance Y. \n" + \
                    "2. CanNotSubmergeIn(Substance X, Substance Y):- VolumeOf(Substance X, Volume Z1), VolumeOf(Substance Y, Volume Z2), BiggerThan(Volume Z1, Volume Z2); \n" + \
                    "If Substance X has a Volume Z1, and Substance Y has a Volume Z2, and Volume Z1 is bigger than Volume Z2, then Substance X cannot submerge in Substance Y. \n" 
    
    example2_input = "Conclusion: CanNotBeAdaptedFrom(Show X, Artwork Y) \n" + \
        "Rules: \n"
    exmaple2_output = "1. CanNotBeAdaptedFrom(Show X, Artwork Y):- ReleasedBefore(Show X, Time Period Z1), PublishedIn(Artwork Y, Time Period  Z2), EarlierThan(Time Period  Z1, Time Period Z2); \n" + \
                    "If Show X was released before Time Period Z1, Artwork Y was published in Time Period Z2, and Time Period Z1 is earlier than Time Period Z2, then Show X cannot be adapted from Artwork Y. \n" + \
                    "2. CanNotBeAdaptedFrom(Show X, Artwork Y):- ProducedDuring(Show X, Time Period Z), WrittenAfter(Artwork Y, Time Period Z); \n" + \
                    "If Show X was produced during Time Period Z and Artwork Y is written after Time Period Z, then Show X cannot be adapted from Artwork Y. \n"
   
    example3_input = f"Conclusion: {conclusion} \n" + \
        f"Rules: \n"

    return [system_prompt, example1_input, exmaple1_output, example2_input, exmaple2_output, example3_input]

all_variables_symbols = [" X", " Y", " Z", " Z1", " Z2"]
variables_token_ids = []
for each in all_variables_symbols:
    variables_token_ids += encoding.encode(each)
variables_logit_dict = {each: 5.0 for each in set(variables_token_ids)}

def get_interaction_logits(concept):
    properties_list = ["Height", "Length", "Weight", "Strength", "Size", "Density", "Volume", 
    "Temperature", "Hardness", "Speed", "BoilingPoint", "MeltingPoint", "Frequency", "Decibel", "Space"]
    properties_list = [" " + each for each in properties_list]
    object_comparison = ["BiggerThan", "SmallerThan", "EarlierThan", "LaterThan", "Under", "Above", "After", "Before"]
    object_comparison = [" " + each for each in object_comparison] + object_comparison[4:]
    
    all_objects = ["Time Period", "Region"] + list(concept.keys())
    all_objects = ["(" + each for each in all_objects] + [" " + each for each in all_objects]
    all_tokens = properties_list + object_comparison + all_objects
    all_tokens = set(all_tokens)
    token_ids = []
    for each in all_tokens:
        token_ids += encoding.encode(each)
    
    logit_dict = {each: 2.0 for each in set(token_ids)}
    logit_dict.update(variables_logit_dict)
    
    return logit_dict

concepts_accessibility = {
    "Animal": ["dog", "cat", "dinosaur"],
    "Plant": ["fruit", "vegatable", "tree"],
    "Food": ["snack", "barbecue", "ingredient", "beverage"],
    "Alcohol": ["wine", "beer", "spirits"],
    "Disease": ["allergy", "cancer", "rhinitis"],
    "Drug": ["antibiotics", "narcotics", "prescription drug"],
    "Natural Phenomenon": ["weather", "natural disaster", "energy"],
    "Condition": ["climate", "symptom", "environment"], 
    "Material": ["fuel", "steel", "plastic", "wood", "stone"],
    "Substance": ["allergen", "gas", "water", "oxygen", "pollen"],
    "Furniture": ["table", "chair", "bed"],
    "Publication": ["album", "song", "book", "discography", "magazine", "poem", "musical work", "written work"],
    "Organization": ["company", "club", "party", "union", "league", "community", "studio"],
    "Facility": ['healthcare facility', 'hospital', 'clinic', 'nursing home', "pharmacy",
                'educational facility', 'university', 'school', 'library', "institution", 'lab',
                'recreation facility', 'park', 'amusement park', 'stadium', 'gym', "museum", 'theater', 
                'production facility', "factory", "farm", "assembly plant", "power plant", "brewery",
                'transport facility', "station", "airport", "railway", "harbour", "port", "publisher",
                'business facility', 'mall', 'restaurant', 'bank', 'market', "shop", "store",
                'administrative facility', 'government', "agency", "authority", "department",
                'religious facility', 'church', 'mosque', 'temple',
                'financial institution', "venue", "landmark", "gallery"],
    "Natural Place": ["mountain", "river", "ocean", "desert", "island", "forest", "volcano", "habitat", "area", "mine"],
    "Event": ["conference", "workshop", "celebration", "race", "activity"],
    "Show": ["movie", "tv show", "drama", "concert", "broadcast", "opera", "cartoon", "comedy"],
    "Artwork": ["photograph", "painting", "sculpture", "architecture", "building"],
    "Job": ["doctor", "teacher", "engineer", "actor", "lawyer", "driver", "profession"],
    "Game": ["sport", "card game", "computer game"],
    "Vehicle": ["car", "aircraft", "ship", "bicycle", "rocket"],
    "Tool": ["container", "weapon", "musical instrument", "kitchen tool", "equipment"],
    "Technology": ["telecommunication", "Internet", "browser", "algorithm", "software"],
    "Electronic Device": ["computer", "phone", "refrigerator", "appliance", "device"],
    "Platform": ["operating system", "social media platform", "streaming media platform", "e-commerce platform", "channel"],
    "Financial Product": ["insurance", "stock", "bond"],
    "Skill": ["knowledge", "language", "recipe", "method", "capability", "experience", "technique", "course", "workexperience"],
    "Authorization": ["credential", "license", "prescription", "identification document", "ticket", "degree", "certification", "qualification", "medicaldegree", "authority"],
    "Legislation": ["policy", "rule", "regulation", "law"],
    "Time Period": ['season', 'times', 'period', "era", "dynasty", "time"],
    "Region": ["country", "city", "town", "location", "state", "province", "place"]
}


from tqdm import tqdm

def interaction_rule_generator(predicate_conc_file, output_rule_file, is_negative=False):
    with open(predicate_conc_file, "r") as r_f:
        interaction_predicate_dict = r_f.readlines()
        print("conclusion number:", len(interaction_predicate_dict))

    cur_logit_dict = get_interaction_logits(concepts_accessibility)
    with open(output_rule_file, 'a') as w_f:
        for each in tqdm(interaction_predicate_dict[:]):
            conclusion = each.strip()
            _, args_variable_list = argument_parsing(conclusion, output_rela=False)
            assert conclusion[1:3] == ". " and conclusion[3:6] == "Can"
            if args_variable_list != ['X', 'Y']:
                continue
            conclusion = conclusion[3:]
            if is_negative:
                conclusion = conclusion[:3] + "Not" + conclusion[3:]
                inputs = get_neg_interaction_premises_input_chat(conclusion)
            else:
                inputs = get_interaction_premises_input_chat(conclusion)
            response = get_chat_response(inputs, temp=0, max_tokens=256, logit_dict=cur_logit_dict)
            w_f.write(response+"\n")   


# edit the rule with variable types, upper variables, not errors
# remove the component in premises same as conclusion
from tqdm import tqdm

def edit_rule(rule_file_name, edit_symbolic_rule_file_name, edit_verbalized_rule_file_name):
    with open(rule_file_name, "r") as rule_r_f:
        rules = rule_r_f.readlines()
    print(rules[:10])
    
    symbolic_rules, verbalized_rules = [], []
    for i in range(len(rules)):
        if i % 2 == 0:
            if ". " not in rules[i]:
                print(i, rules[i])
            assert ". " in rules[i]
            if ":- " not in rules[i]:
                print(i, rules[i])
            assert ":- " in rules[i]
            symbolic_rules.append(rules[i].split(". ")[1])
        else:
            if ". " in rules[i].strip():
                print(i, rules[i])
            assert "1. " not in rules[i].strip() and "2. " not in rules[i].strip()
            # assert ". " in rules[i].strip()
            if not (":- " not in rules[i] and "if" in rules[i].lower()):
                print(i, rules[i])
            assert ":- " not in rules[i] and "if" in rules[i].lower() #and "If" in rules[i] #only if
            # verbalized_rules.append(rules[i].split(". ")[1])
            verbalized_rules.append(rules[i].strip())
    assert len(symbolic_rules) == len(verbalized_rules)
    print(len(symbolic_rules), len(verbalized_rules))
    
    edit_symbolic_rules = []
    edit_verbalized_rules = []
    edit_num_1 = 0
    edit_num_2 = 0
    for n, each_rule in tqdm(enumerate(symbolic_rules)):
        each_rule = each_rule.replace("PersonX", "Person X").replace("PersonY", "Person Y").replace("Personz1", "Person Z1").replace("Personz2", "Person Z2")
        next_rule = False
        rule = each_rule.strip()

        assert ":- " in rule
        conclusion, premises = rule.split(":- ")
        _, conc_args_types, conc_args_variables = argument_parsing(conclusion, output_rela=True)
        if None in conc_args_types:
            print("hh", each_rule)
        assert None not in conc_args_types
        if not ("X" in conc_args_variables and "Y" in conc_args_variables):
            print("hh", each_rule)
            continue
        assert "X" in conc_args_variables and "Y" in conc_args_variables

        variable_type_dict = {}
        variable_type_dict[conc_args_variables[0]] = conc_args_types[0]
        variable_type_dict[conc_args_variables[1]] = conc_args_types[1]
        
        premises_list = premises.split("),")
        premises_list = [_.strip() for _ in premises_list]
        for i in range(len(premises_list)):
            if i < len(premises_list) - 1:
                premises_list[i] = premises_list[i] + ")"
            elif premises_list[i][-1] == ";" or premises_list[i][-1] == ".":
                premises_list[i] = premises_list[i][:-1]
        
        new_premise_list = [] 
        for each_premise in premises_list:
            cur_rela, cur_args_types, cur_args_variables = argument_parsing(each_premise, output_rela=True)
            if len(cur_args_variables) == 0:
                next_rule = True
                break
            cur_args_variables = [each.upper() if each is not None else each for each in cur_args_variables]
            new_premise = cur_rela + "("
            for j in range(len(cur_args_types)):
                if cur_args_types[j] is None:
                    if cur_args_variables[j] is not None:
                        cur_args_types[j] = variable_type_dict.get(cur_args_variables[j], None)
                else:
                    if cur_args_variables[j] is not None and cur_args_variables[j] not in variable_type_dict:
                        variable_type_dict[cur_args_variables[j]] = cur_args_types[j]
                new_premise += (str(cur_args_types[j]) + " " + str(cur_args_variables[j])).replace("None", "").strip()
                if j < len(cur_args_types) - 1:
                    new_premise += ", "
            new_premise += ")"
            new_premise_list.append(new_premise)
        
        if not next_rule:
            new_rule = conclusion + ":- " + ", ".join(new_premise_list) + ";"

            if new_rule not in edit_symbolic_rules:
                edit_symbolic_rules.append(new_rule)
                edit_verbalized_rules.append(verbalized_rules[n].strip())
        else:
            pass
        
    print(edit_num_1, edit_num_2)
    print(len(edit_symbolic_rules), len(edit_verbalized_rules))

    with open(edit_symbolic_rule_file_name, 'w') as w_f_1:
        for each in edit_symbolic_rules:
            w_f_1.write(each+"\n")
    with open(edit_verbalized_rule_file_name, 'w') as w_f_2:
        for each in edit_verbalized_rules:
            w_f_2.write(each+"\n")
                

def primitive_filter(rule_file, verbalized_rule_file, write_file_name, write_vb_file_name, concepts):
    with open(rule_file, "r") as r_f:
        rules = r_f.readlines()
    with open(verbalized_rule_file, "r") as v_r_f:
        verbalized_rules = v_r_f.readlines()
    print(len(rules), len(verbalized_rules))

    def find_super_concepts(type):
        for each in concepts:
            if type.lower() in concepts[each]:
                return each
        return None

    constraint_rules = []
    constraint_verbalized_rules = []
    properties_list = ["Age", "Price", "Money", "Height", "Length", "Weight", "Strength", "Size", "Density", "Volume", 
        "Temperature", "Hardness", "Speed", "BoilingPoint", "MeltingPoint", "Frequency", "Decibel", "Space"]
    candidate_concepts = list(concepts.keys()) + ["Person"] + properties_list
    for n, each_rule in enumerate(rules):
        each_rule = each_rule.strip()
        conclusion, premises_list = parse_sentence_to_conclusion_premise(each_rule)
        conc_args_type_list, _ = argument_parsing(conclusion, output_rela=False)

        replace_types = []
        jump_to_next = False
        for each_premise in [conclusion] + premises_list:
            args_type_list, _ = argument_parsing(each_premise, output_rela=False)
            if args_type_list[0] not in candidate_concepts:
                super_type = find_super_concepts(args_type_list[0])
                if super_type is not None and super_type != args_type_list[1] and super_type not in conc_args_type_list:
                    if [args_type_list[0], super_type] not in replace_types:
                        replace_types.append([args_type_list[0], super_type])
                else:
                    jump_to_next = True
                    break
            elif len(args_type_list) > 1 and args_type_list[1] not in candidate_concepts:
                super_type = find_super_concepts(args_type_list[1])
                if super_type is not None and super_type != args_type_list[0] and super_type not in conc_args_type_list:
                    if [args_type_list[1], super_type] not in replace_types:
                        replace_types.append([args_type_list[1], super_type])
                else:
                    jump_to_next = True
                    break
        if jump_to_next:
            continue
        else:
            each_verbalized_rule = verbalized_rules[n].strip()
            for each_pair in replace_types:
                each_rule = each_rule.replace(each_pair[0], each_pair[1])
                each_verbalized_rule = each_verbalized_rule.replace(each_pair[0], each_pair[1]).replace(each_pair[0][0].lower() + each_pair[0][1:], each_pair[1])
        if each_rule not in constraint_rules:
            constraint_rules.append(each_rule)
            constraint_verbalized_rules.append(each_verbalized_rule)
    print(len(constraint_rules), len(constraint_verbalized_rules))

    with open(write_file_name, 'w') as w_f:
        for each in constraint_rules:
            w_f.write(each+"\n")
    with open(write_vb_file_name, 'w') as w_f_2:
        for each in constraint_verbalized_rules:
            w_f_2.write(each+"\n")


def negative_premise_filter(rule_file, verbalized_rule_file, write_file_name, write_vb_file_name):
    with open(rule_file, "r") as r_f:
        rules = r_f.readlines()
    with open(verbalized_rule_file, "r") as v_r_f:
        verbalized_rules = v_r_f.readlines()
    print(len(rules), len(verbalized_rules))

    positive_premise_rules = []
    positive_premise_verbalized_rules = []
    for n, each_rule in enumerate(rules):
        each_rule = each_rule.strip()
        assert ":- " in each_rule
        conclusion, premises = each_rule.split(":- ")

        if "not" not in premises.lower() and "never" not in premises.lower():
            positive_premise_rules.append(each_rule)
            positive_premise_verbalized_rules.append(verbalized_rules[n].strip())

    print(len(positive_premise_rules), len(positive_premise_verbalized_rules))

    with open(write_file_name, 'w') as w_f:
        for each in positive_premise_rules:
            w_f.write(each+"\n")
    with open(write_vb_file_name, 'w') as w_f_2:
        for each in positive_premise_verbalized_rules:
            w_f_2.write(each+"\n")


def get_verbalized_critic_input(each_rule):    
    input = "True or False? Please predict whether the input rule is accurate or not according to commonsense knowledge in realistic scenarios, and also explain why. \n\nExamples:\n" + \
            "Input: If Person X was born in Season Z and Plant Y blooms in the same Season Z, then Person X can access Plant Y. \n" + \
            "Output: False. Because the season of a person's birth and the blooming season of a plant has no logical connection. \n" + \
            "Input: If Person X has an Age Z1 and Vehicle Y requires an Age above Z2 for driving, with Age Z1 being greater than Age Z2, then Person X can drive Vehicle Y. \n" + \
            "Output: True. Because driving vehicle has a minimum age requirement. \n" + \
            "Input: If Person X has Capital Z1 and the minimum capital requirement for establishing Organization Y is Capital Z2, and Capital Z1 is bigger than Capital Z2, then Person X can establish Organization Y. \n" + \
            "Output: False. Because person can not have a capital and capital is not suitable for value comparison. \n" + \
            "Input: If Person X is allergic to Material Z1, and Clothing Y contains Material Z1, then Person X cannot wear Clothing Y. \n" + \
            "Output: True. Because person should avoid contact with allergenic substances. \n\n" + \
            "Input: " + each_rule + "\n" + \
            "Output:\n"
    return input


# classify the verbalized rules via GPT-4
def rule_critic(rule_file, verbalized_rule_file, write_file_name, write_vb_file_name):
    with open(rule_file, "r") as r_f:
        rules = r_f.readlines()
    with open(verbalized_rule_file, "r") as v_r_f:
        verbalized_rules = v_r_f.readlines()
    print(len(rules), len(verbalized_rules))
    
    verbalized_label_list = []
    for each_rule in tqdm(verbalized_rules):
        v_critic_input = get_verbalized_critic_input(each_rule.strip())
        v_response = get_LLama_response(v_critic_input, max_tokens=40, temp=0) 
        verbalized_label_list.append(v_response)
    print(verbalized_label_list[:20])

    print(len(verbalized_label_list))
    print(len(rules), len(verbalized_rules))

    critic_symbolic_rules = []
    critic_verbalized_rules = []
    for i in range(len(verbalized_label_list)):
        if "True" in verbalized_label_list[i]:
            if rules[i].strip() not in critic_symbolic_rules:
                critic_symbolic_rules.append(rules[i].strip())
                critic_verbalized_rules.append(verbalized_rules[i].strip())
    print(len(critic_symbolic_rules), len(critic_verbalized_rules), len(verbalized_label_list)-len(critic_symbolic_rules))

    with open(write_file_name, 'w') as w_f:
        for each in critic_symbolic_rules:
            w_f.write(each+"\n")
    with open(write_vb_file_name, 'w') as w_f_2:
        for each in critic_verbalized_rules:
            w_f_2.write(each+"\n")


def get_diversify_input(premise):
    query_rule_prompt = f"According to commonsense knowledge in realistic scenarios, please generate 3 logical rules in Prolog to describe the conclusion of the given premise. \n" +\
        f"The conclusion should contain two variables X and Y. \n\nExamples:\n" +\
        f"Premise: BornIn(Person X, Region Y) \n" + \
        f"Rules: \n" + \
        f"1. KnowsCulture(Person X, Region Y):- BornIn(Person X, Region Y); \n" + \
        f"If Person X were born in Region Y, then Person X knows the culture of Region Y. \n" + \
        f"2. HasNationality(Person X, Region Y):- BornIn(Person X, Region Y); \n" + \
        f"If Person X were born in Region Y, then Person X has the nationality of Region Y. \n" + \
        f"3. EligibleForPassport(Person X, Region Y):- BornIn(Person X, Region Y); \n" + \
        f"If Person X was born in Region Y then Person X is eligible for a passport from Region Y. \n\n" + \
        f"Premise: {premise}\n" + \
        f"Rules: \n"
    return query_rule_prompt

def get_diversify_input_reverse(premise):
    query_rule_prompt = f"According to commonsense knowledge in realistic scenarios, please generate 3 logical rules in Prolog to describe the premise of the given conclusion. \n" +\
        f"Each rule should contain only one premise with two variables X and Y. \n\nExamples:\n" +\
        f"Conclusion: Have(Person X, Skill Y) \n" + \
        f"Rules: \n" + \
        f"1. Have(Person X, Skill Y):- Learned(Person X, Skill Y); \n" + \
        f"If Person X learned Skill Y, then Person X has Skill Y. \n" + \
        f"2. Have(Person X, Skill Y):- Inherit(Person X, Skill Y); \n" + \
        f"If Person X inherits Skill Y, then Person X has Skill Y. \n" + \
        f"3. Have(Person X, Skill Y):- Acquire(Person X, Skill Y); \n" + \
        f"If Person X acquires Skill Y, then Person X has Skill Y. \n\n" + \
        f"Conclusion: {premise}\n" + \
        f"Rules: \n"
    return query_rule_prompt



# take each premise as conclusion
from tqdm import tqdm
def get_premise_extension(file_name, extension_file):
    not_extend_keywords_list = ["than", "same", "locatedin", "different", "similar", ]
    with open(file_name, "r") as r_f:
        rules = r_f.readlines()
        
    def get_sub_conclusions(each_rule):
        premises_list = parse_sentence_to_conclusion_premise(each_rule)[1]
        
        cur_sub_conclusions = []
        for each_premise in premises_list:
            each_rela, each_args_types, each_args_variables = argument_parsing(each_premise, output_rela=True)
            assert len(each_args_variables) == 2
            each_sub_conclusion = each_rela + "(" + each_args_types[0] + " X, " + each_args_types[1] + " Y)"
            each_sub_conclusion = each_sub_conclusion[0].upper() + each_sub_conclusion[1:]
            cur_sub_conclusions.append(each_sub_conclusion)
        return cur_sub_conclusions

    sub_conclusions = []
    for each_rule in rules:
        cur_sub_conclusions = get_sub_conclusions(each_rule)
        for each in cur_sub_conclusions:
            skip_conc = False
            for word in not_extend_keywords_list:
                if word in each.lower():
                    skip_conc = True
                    break
            if not skip_conc:
                if each not in sub_conclusions: 
                    sub_conclusions.append(each)
    print(len(sub_conclusions))

    with open(extension_file, "w") as w_f:   
        for each_sub_conclusion in tqdm(sub_conclusions):
            input = get_diversify_input_reverse(each_sub_conclusion)
            response = get_LLama_response(input, max_tokens=256, temp=0)
            w_f.write(response+"\n")

# take each conclusion as premise
from tqdm import tqdm
def get_conclusion_extension(file_name, extension_file):
    with open(file_name, "r") as r_f:
        rules = r_f.readlines()

    sub_premises = []
    for each_rule in rules:
        cur_sub_premise = parse_sentence_to_conclusion_premise(each_rule)[0]
        assert "X" in cur_sub_premise and "Y" in cur_sub_premise
        if cur_sub_premise not in sub_premises: 
            sub_premises.append(cur_sub_premise)
    print(len(sub_premises))

    with open(extension_file, "a") as w_f:
        for each_sub_premise in tqdm(sub_premises[1492:]):
            input = get_diversify_input(each_sub_premise)
            response = get_LLama_response(input, max_tokens=256, temp=0)
            w_f.write(response+"\n")

def substitute_extension(extension_file):
    with open(extension_file, "r") as r_f:
        extension_rules = r_f.readlines()
    
    extension_dict = {}
    for each_rule in extension_rules:
        conclusion, premises_list = parse_sentence_to_conclusion_premise(each_rule)
        assert len(premises_list) == 1

        if conclusion not in extension_dict:
            extension_dict[conclusion] = premises_list
        else:
            if premises_list[0] not in extension_dict[conclusion]:
                extension_dict[conclusion] += premises_list
    print(len(extension_dict))
    return extension_dict

def substitute_conclusion_extension(extension_file):
    with open(extension_file, "r") as r_f:
        extension_rules = r_f.readlines()
    
    extension_dict = {}
    for each_rule in extension_rules:
        conclusion, premises_list = parse_sentence_to_conclusion_premise(each_rule)
        assert len(premises_list) == 1

        if premises_list[0] not in extension_dict:
            extension_dict[premises_list[0]] = [conclusion]
        else:
            if conclusion not in extension_dict[premises_list[0]]:
                extension_dict[premises_list[0]].append(conclusion)
    print(len(extension_dict))
    return extension_dict


from tqdm import tqdm
def substitute_all_components(run_file_name, extension_file, conc_extension_file, extend_rule_file):
    extension_dict = substitute_extension(extension_file)
    extension_record = {}
    for each in extension_dict:
        extension_record[each] = 0
    conc_extension_dict = substitute_conclusion_extension(conc_extension_file)
    conc_extension_record = {}
    for each in conc_extension_dict:
        conc_extension_record[each] = 0

    with open(run_file_name, "r") as r_f:
        rules = r_f.readlines()

    extend_rules = []
    for i, each_rule in tqdm(enumerate(rules)):
        conclusion, premises_list = parse_sentence_to_conclusion_premise(each_rule)

        for premise_index in range(len(premises_list)):  
            each_rela, each_args_types, each_args_variables = argument_parsing(premises_list[premise_index], output_rela=True)
            type_variabels_dict = {}
            type_variabels_dict[each_args_types[0]] = each_args_variables[0]
            type_variabels_dict[each_args_types[1]] = each_args_variables[1]
        
            each_key = each_rela + "(" + each_args_types[0] + " X, " + each_args_types[1] + " Y)"
            each_key = each_key[0].upper() + each_key[1:]
            if each_key in extension_dict:
                for prem_extend_index in range(len(extension_dict[each_key])):
                    each_prem = extension_dict[each_key][prem_extend_index]
                    each_prem_rela, each_prem_args_types, each_prem_args_variables = argument_parsing(each_prem, output_rela=True)
                    each_prem_edit_variable = f"{each_prem_rela}({each_prem_args_types[0]} {type_variabels_dict[each_prem_args_types[0]]}, {each_prem_args_types[1]} {type_variabels_dict[each_prem_args_types[1]]})"
                    new_premises_list = premises_list[:premise_index] + [each_prem_edit_variable,] + premises_list[premise_index+1:]
                    new_rule = conclusion + ":- " + ", ".join(new_premises_list) + ";"
                    if new_rule not in extend_rules:
                        extend_rules.append(new_rule)
        
        if conclusion in conc_extension_dict:
            for conc_extend_index in range(len(conc_extension_dict[conclusion])):
                each_conc = conc_extension_dict[conclusion][conc_extend_index]
                new_rule = each_conc + ":- " + ", ".join(premises_list) + ";"
                if new_rule not in extend_rules:
                    extend_rules.append(new_rule)

                for _ in range(2):
                    new_premises_list = []
                    for i in range(len(premises_list)):
                        each_rela, each_args_types, each_args_variables = argument_parsing(premises_list[i], output_rela=True)
                        assert len(each_args_variables) == 2 and len(each_args_types) == 2
                        type_variabels_dict = {}
                        type_variabels_dict[each_args_types[0]] = each_args_variables[0]
                        type_variabels_dict[each_args_types[1]] = each_args_variables[1] 
                        each_key = each_rela + "(" + each_args_types[0] + " X, " + each_args_types[1] + " Y)"
                        each_key = each_key[0].upper() + each_key[1:]

                        if each_key in extension_dict:
                            extend_index = extension_record[each_key]
                            each_prem = extension_dict[each_key][extend_index]
                            each_prem_rela, each_prem_args_types, each_prem_args_variables = argument_parsing(each_prem, output_rela=True)
                            each_prem_edit_variable = f"{each_prem_rela}({each_prem_args_types[0]} {type_variabels_dict[each_prem_args_types[0]]}, {each_prem_args_types[1]} {type_variabels_dict[each_prem_args_types[1]]})"
                            new_premises_list.append(each_prem_edit_variable)
                            extension_record[each_key] = (extension_record[each_key] + 1) % len(extension_dict[each_key])
                        else:
                            new_premises_list.append(premises_list[i])

                    new_rule = each_conc + ":- " + ", ".join(new_premises_list) + ";"
                    if new_rule not in extend_rules:
                        extend_rules.append(new_rule)        

    print(len(extend_rules))
    with open(extend_rule_file, "w") as w_f:
        for each in tqdm(extend_rules):
            w_f.write(each+"\n")


def filter_diversified_rules(extend_rule_file, filter_extend_rule_file, thres=3):
    with open(extend_rule_file, "r") as r_f:
        diversified_rules = r_f.readlines()
    print(len(diversified_rules))

    filtered_rules = []
    for each_rule in tqdm(diversified_rules):
        each_rule = each_rule.strip()
        conclusion, premises_list = parse_sentence_to_conclusion_premise(each_rule)

        conc_rela, _, _ = argument_parsing(conclusion, output_rela=True)
        premise_rela_list = []
        for i in range(len(premises_list)):
            each_rela, each_args_types, each_args_variables = argument_parsing(premises_list[i], output_rela=True)
            assert len(each_args_variables) == 2 and len(each_args_types) == 2
            premise_rela_list.append(each_rela.lower())
        _, max_similarity = get_most_similar_premise(conc_rela.lower(), premise_rela_list)
        if max_similarity <= thres:
            filtered_rules.append(each_rule)
    print(len(filtered_rules))

    with open(filter_extend_rule_file, "w") as w_f:
        for each in tqdm(filtered_rules):
            w_f.write(each+"\n")


def get_whole_verbalize_input(each_premise):
    input = "Please verbalize each input rule into a natural langauge statement in 'if-then' format. \n\nExamples:\n" + \
            "Rule:\n" + \
            "CanRequest(Person X, Authorization Y):- Have(Person X, Age Z1), RequireMinimumAge(Authorization Y, Age Z2), BiggerThan(Age Z1, Age Z2);\n" + \
            "Statement:\n" + \
            "If Person X has Age Z1 and the minimum age requirement for requesting Authorization Y is Age Z2, Age Z1 is bigger than Age Z2, then Person X can request Authorization Y. \n" + \
            "Rule:\n" + \
            "CanRepair(Person X, Electronic Device Y):- Master(Person X, Skill Z2), RequiredForRepairing(Skill Z2, Electronic Device Y);\n" + \
            "Statement:\n" + \
            "If Person X has mastered Skill Z2 and Skill Z2 is required for repairing Electronic Device Y, then Person X can repair Electronic Device Y.\n\n" + \
            "Rule:\n" + \
            each_premise + "\n" + \
            "Statement:\n"
    return input

def verbalized_extended_rules(file_name, write_verbalized_rule_file):
    with open(file_name, "r") as r_f:
        rules = r_f.readlines()

    with open(write_verbalized_rule_file, "w") as w_f:
        for each_rule in tqdm(rules):
            input = get_whole_verbalize_input(each_rule.strip())
            response = get_LLama_response(input, max_tokens=100, temp=0, model="gpt-3.5-turbo")
            w_f.write(response+"\n")


def remove_concept(text, concepts):
    for each in list(concepts.keys()) + ["Person", "Region"]:
        if each+" " in text:
            text = text.replace(each+" ", "")
    text = text.replace("X ", "").replace("X", "").replace("Y ", "").replace("Y", "")
    return text

def filter_diversified_verblized_rules(rule_file, vb_rule_file, filter_file_name, filter_vb_file_name, concepts):
    with open(rule_file, "r") as r_f:
        rules = r_f.readlines()
    with open(vb_rule_file, "r") as r_f:
        vb_rules = r_f.readlines()
    print(len(rules), len(vb_rules))

    filtered_rules = []
    filtered_vb_rules = []
    for n, each_rule in tqdm(enumerate(vb_rules)):
        each_rule = each_rule.strip()
        assert each_rule[:3] == "If " and ", then " in each_rule
        premise, conclusion = each_rule[3:].split(", then ")
        premise = remove_concept(premise, concepts).lower()
        conclusion = remove_concept(conclusion, concepts).lower()

        similarity = find_lcseque(premise, conclusion)
        if similarity <= 10: # multiple 15 single 12
            filtered_rules.append(rules[n].strip())
            filtered_vb_rules.append(each_rule)
    print(len(filtered_rules), len(filtered_vb_rules))

    with open(filter_file_name, "w") as w_f:
        for each in tqdm(filtered_rules):
            w_f.write(each+"\n")
    with open(filter_vb_file_name, "w") as w_f:
        for each in tqdm(filtered_vb_rules):
            w_f.write(each+"\n")


def prepara_annotation_data(verbalized_rule_file, annotation_file):
    with open(verbalized_rule_file, "r") as v_r_f:
        verbalized_rules = v_r_f.readlines()
        print(len(verbalized_rules))

    selected_index = range(1000, len(verbalized_rules)) 

    data_list = []
    for i in selected_index:
        each_data = {}
        each_data['id'] = i
        each_rule = verbalized_rules[i].strip()
        assert each_rule[:3] == "If " and ", then " in each_rule
        premise, conclusion = each_rule[3:].split(", then ")
        premise = premise + "."
        each_data['premise'] = premise
        each_data['conclusion'] = conclusion
        data_list.append(each_data)
    print(len(data_list))    

    import csv
    with open(annotation_file, "w") as csv_f:
        writer = csv.writer(csv_f)
        writer.writerow(['entryid', 'premise', 'conclusion'])

        for each in data_list:
            writer.writerow([each['id'], each['premise'], each['conclusion']])


import csv
from tqdm import tqdm
def process_annotation_data(rule_file, verbalized_rule_file, result_file, write_file_name, write_verbalized_file_name):
    with open(rule_file, "r") as v_r_f:
        rules = v_r_f.readlines()
        print(len(rules))
    with open(verbalized_rule_file, "r") as v_r_f:
        verbalized_rules = v_r_f.readlines()
        print(len(verbalized_rules))

    all_data = []
    with open(result_file) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            all_data.append(row)
    print(len(all_data)-1)
    print("worker_id", all_data[0].index("WorkerId"))

    annotation_dict_correct = {}
    annotation_dict_common_premise = {}
    annotation_dict_common_conc = {}
    for i in range(1, len(all_data)):
        if all_data[i][15] != 'A3MTHVR1EJ8LMM':
            entry_id = all_data[i][27]
            if entry_id not in annotation_dict_correct:
                annotation_dict_correct[entry_id] = [all_data[i][32]]
            else:
                annotation_dict_correct[entry_id].append(all_data[i][32]) 
            
            if entry_id not in annotation_dict_common_premise:
                annotation_dict_common_premise[entry_id] = [all_data[i][33]]
            else:
                annotation_dict_common_premise[entry_id].append(all_data[i][33]) 
            
            if entry_id not in annotation_dict_common_conc:
                annotation_dict_common_conc[entry_id] = [all_data[i][34]]
            else:
                annotation_dict_common_conc[entry_id].append(all_data[i][34]) 
    print(len(annotation_dict_correct), len(annotation_dict_common_premise), len(annotation_dict_common_conc))

    all_annotate_ids = []
    filter_ids_1 = []
    agree_num = 0
    for each in annotation_dict_correct:
        if len(annotation_dict_correct[each]) < 3:
            all_annotate_ids.append(each)

        entaiment_count = annotation_dict_correct[each].count("2")
        premise_count = annotation_dict_common_premise[each].count("2")
        conc_count = annotation_dict_common_conc[each].count("2")
        if premise_count >= 3 and conc_count >= 3 and entaiment_count >= 3:   
            filter_ids_1.append(each)
        
        if len(set(annotation_dict_correct[each])) == 1:
            agree_num += 1
    correct_index = sorted(filter_ids_1)
    print(len(correct_index))
    print("all_annotate_ids", all_annotate_ids)
    print("Agreement", agree_num, agree_num/len(annotation_dict_correct))

    with open(write_file_name, 'w') as w_f:
        for id in correct_index:
            w_f.write(rules[int(id)].strip()+"\n")
    with open(write_verbalized_file_name, 'w') as w_f:
        for id in correct_index:
            w_f.write(verbalized_rules[int(id)].strip()+"\n")


def remove_concept(text, concepts):
    for each in list(concepts.keys()) + ["Person", "Region"]:
        if each+" " in text:
            text = text.replace(each+" ", "")
    text = text.replace("X ", "").replace("X", "").replace("Y ", "").replace("Y", "")
    return text

def post_filter_repetitive_rules(rule_file, vb_rule_file, filter_file_name, filter_vb_file_name, concepts, thres=3):
    with open(rule_file, "r") as r_f:
        rules = r_f.readlines()
    with open(vb_rule_file, "r") as r_f:
        vb_rules = r_f.readlines()
    print(len(rules), len(vb_rules))

    filtered_rules = []
    filtered_vb_rules = []
    for n, each_rule in tqdm(enumerate(vb_rules)):
        each_symbolic_rule = rules[n].strip()
        symbolic_conc, premises_list = parse_sentence_to_conclusion_premise(each_symbolic_rule)
        conc_rela, _, _ = argument_parsing(symbolic_conc, output_rela=True)
        premise_rela_list = []
        for i in range(len(premises_list)):
            each_rela, each_args_types, each_args_variables = argument_parsing(premises_list[i], output_rela=True)
            assert len(each_args_variables) == 2 and len(each_args_types) == 2
            premise_rela_list.append(each_rela.lower())
        _, max_similarity = get_most_similar_premise(conc_rela.lower(), premise_rela_list)

        each_rule = each_rule.strip()
        assert each_rule[:3] == "If " and ", then " in each_rule
        premise, conclusion = each_rule[3:].split(", then ")
        premise = remove_concept(premise, concepts).lower()
        conclusion = remove_concept(conclusion, concepts).lower()

        similarity = find_lcseque(premise, conclusion)
        if similarity <= 10 and max_similarity <= thres:
            filtered_rules.append(each_symbolic_rule)
            filtered_vb_rules.append(each_rule)
    print(len(filtered_rules), len(filtered_vb_rules))

    with open(filter_file_name, "w") as w_f:
        for each in tqdm(filtered_rules):
            w_f.write(each+"\n")
    with open(filter_vb_file_name, "w") as w_f:
        for each in tqdm(filtered_vb_rules):
            w_f.write(each+"\n")