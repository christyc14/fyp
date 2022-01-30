from typing import Dict, List, Union
from zlib import DEF_BUF_SIZE
import json_lines
import numpy as np

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
    # print(df)
    
# creating a dictionary of ingredients, need to clean up ingredient (eg with *s, multiple names for water)
    ingred_index = {}
    igs = []
    index = 0
    i = 0
    for item in df:
        igs.append(item['ingredients'])
        for i in range(len(item['ingredients'])):
            if item['ingredients'][i] not in ingred_index:
                ingred_index[index] = item['ingredients'][i]
            index +=1
    print(ingred_index)
# mapping encodings to items
    X = len(df)
    Y = len(ingred_index)

    big_matrix = np.zeros((X,Y))

    def encoder(ing):
        a = np.zeros(Y)
        for i in ing:
            #get the index for each ingredient
            # index = ingred_index[i]
            index = list(ingred_index.keys())[list(ingred_index.values()).index(i)]
            # print(index)
            a[index] = 1
        return a 
    
    i = 0
    print(len(igs))
    for ingreds in igs:
        big_matrix[i, :] = encoder(ingreds)
        i += 1

#tsne to reduce dimensionality

#can add the multiple products later

print(recommender('MOISTURISER'))