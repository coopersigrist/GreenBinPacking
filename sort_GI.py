import random

with open("./Data/GI_3.txt", "r") as file:
    lines = file.readlines()
    lines = list(map(lambda x: str(int(x)/1000000)+"\n", lines))

random.shuffle(lines)

with open("./Data/GI.txt", "w") as file:
    file.writelines(lines)