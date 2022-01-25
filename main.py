from typing import Dict, List, Union
import json_lines

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
def recommender(option):
    
# tokenize