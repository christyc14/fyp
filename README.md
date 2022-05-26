# Two Skincare Product Recomender Algorithms
One is a hybrid recommender, and the other is a machine learning recommender. Use the ```sephora``` branch.
## Run locally
``` cd <path_to>/fyp/ ```\
To scrape:\
1. Start Docker
2. ``` docker run -p 8050:8050 scrapinghub/splash```
3. VPN to US
4. ```cd cbscraper``` 
5. ```scrapy crawl productbot -o <file_name>.jsonlines```\
To run the testing form:
```streamlit run form.py```