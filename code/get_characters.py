characters = []

def add_chars(filename):
    with open(filename) as f:
        for line in f:
            for c in line:
                if c not in characters:
                    characters.append(c)

add_chars("../data/train.context")
add_chars("../data/train.question")
print(characters)
