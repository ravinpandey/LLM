import random

rules = [
    ["S","NP VP"],
    ["Q","who does NP V # NP $"],

    ["NP","DET N"],
    ["NP","DET A N"],

    ["A","dirty"],
    ["A","clean"],
    ["A","rich"],
    ["A","poor"],
    ["A", "young"],
    ["A", "old"],

    ["DET","the"],
    ["DET","a"],

    ["N","dog"],
    ["N","cat"],
    ["N", "frog"],
    ["N", "rabbit"],

    ["VP","V NP"],

    ["V","hits"],
    ["V","bites"],
    ["V","knows"],
    ["V","loves"],
    ["V","hates"],
    ["V","admires"],
]

def random_sentence(sentence):
    while True:
        possible_rules = []
        for rule in rules:
            if sentence.find(' ' + rule[0] + ' ')>-1:
                possible_rules.append(rule)

        if not possible_rules:
            break
        rule = random.choice(possible_rules)
        sentence = sentence.replace(rule[0],rule[1],1)
    return sentence.strip() + " \n"


with open("stories.txt", "w") as my_file:
    for i in range(1,1000):
        my_file.write(f"{random_sentence(' S ')}")

with open("stories-sft.txt", "w") as my_file:
    for i in range(1,1000):
        my_file.write(f"{random_sentence(' Q ')}")


