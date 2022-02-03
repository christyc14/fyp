from typing import Dict, List, Union
from zlib import DEF_BUF_SIZE
import json_lines
import numpy as np
import re

if __name__ == "__main__":
    print("Hello world!")

products: List[Dict[str, Union[str, List[str]]]] = []
# input data into List
with open('cbscraper/product_urls.jsonlines', 'rb') as f:
    for item in json_lines.reader(f):
        products.append(item)

# option: category of product, eg cleanser
option = []
seen = set()
for item in products:
    if item['category'] not in seen:
        seen.add(item['category'])
        option.append(item['category'])
print(option)
# filter data by given option

def recommender(opt):
    df = []
    for item in products:
        if opt == item['category']:
            df.append(item)
    
# creating a dictionary of ingredients, and cleaning up the dataset
    # ingred_index = set()
    # igs = []
    # # index = 0
    # i = 0
    # for item in df:
    #     igs.append(item['ingredients'])
    #     for i in range(len(item['ingredients'])):
    #         ingr = re.sub(r"\(([^)]*)\)|(([0-9]\d{0,2}(\.\d{1,3})*(,\d+)?)(%|mg|units))|(<\/?i>)|(\/.+)|(\\.+)|\[([^\]]*)\]", '', item['ingredients'][i])
    #         if ingr.lower() == "water" or ingr.lower() == "aqua" or ingr.lower() == "eau":
    #             ingr = "Water"
    #         if ingr not in ingred_index:
    #             ingred_index.add(ingr)
    #         # index +=1
    output = []
    for item in df:
        ings = []
        for i in range(len(item['ingredients'])):
            ingr = re.sub(r"\(([^)]*)\)|(([0-9]\d{0,2}(\.\d{1,3})*(,\d+)?)(%|mg|units))|(<\/?i>)|(\/.+)|(\\.+)|\[([^\]]*)\]", '', item['ingredients'][i])
            if ingr.lower() == "water" or ingr.lower() == "aqua" or ingr.lower() == "eau":
                ingr = "Water"
            ings.append(ingr)
        output.append(ings)
    
    enc = OneHotEncoder(handle_unknown='ignore')
# mapping encodings to items
    X = len(df)
    Y = len(ingred_index)

    big_matrix = np.zeros((X,Y))

    def encoder(ing):
        a = np.zeros(Y)
        for i in ing:
            #get the index for each ingredient
            index = list(ingred_index.keys())[list(ingred_index.values()).index(i)]
            a[index] = 1
        return a 
    
    i = 0
    for ingreds in igs:
        big_matrix[i, :] = encoder(ingreds)
        i += 1

#tsne to reduce dimensionality

#can add the multiple products later

print(recommender('MOISTURISER'))
