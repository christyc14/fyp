# Two Skincare Product Recomender Algorithms
One is a hybrid recommender, and the other is a machine learning recommender. Use the ```sephora``` branch.
### Run locally
``` cd <path_to>/fyp/ ```

### To scrape:
1. Start Docker

2. ``` docker run -p 8050:8050 scrapinghub/splash```

3. VPN to US

4. ```cd cbscraper``` 

5. ```scrapy crawl productbot -o <file_name>.jsonlines```

### To run the testing form:
```streamlit run form.py```

### To run the hybrid recommender:
```streamlit run hybrid.py```

### To run the ML recommender:
```streamlit run ml.py```

The ML model was trained using Google Colab, it is available [here](https://colab.research.google.com/drive/1MedeIX4qDW2akEfw2tL6m0SPi6N0-gMW?usp=sharing)

### To run on VM continuously:
```nohup streamlit form.py &``` 

This pipes all terminal output to a file called ```nohup.out```

### To update stuff on VM (using GCP):
1. ```gcloud compute ssh --zone "us-west4-b" "instance-2"  --project "fyp-proj-350515"```

2. ```cd fyp```

3. ```git pull```

4. Run as usual.
