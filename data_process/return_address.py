import os

def return_address(mainAddress, target, mode = "method"):
    subPaths = [os.path.join(mainAddress, i) for i in os.listdir(mainAddress) if i != '.DS_Store']
    subNames = [j for j in os.listdir(mainAddress) if j != '.DS_Store']
    for index, i in enumerate(subNames):
        if mode == "method":
            indicator = i
        elif mode == "case" or mode == "miss":
            indicator = i[4:]
        elif mode == "full_file" or mode == "miss_file":
            indicator = i.split('.')[0]

        if target == indicator:
            return subPaths[index]
        else:
            continue