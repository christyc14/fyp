from ast import Dict
from enum import Enum, auto
import json
import re
from scrapy import Request, Spider

BASE_URL = "https://www.sephora.com"
NUM_PAGES = 3
EMPTY_DESCRIPTION = "\n\n                              "


class Category(Enum):
    CLEANSER = "face-wash-facial-cleanser"
    TONER = "facial-toner-skin-toner"
    MOISTURISER = "moisturizer-skincare"
    SERUM = "face-serum"
    MASKS = "exfoliating-scrub-exfoliator"
    SPF = "face-sunscreen"

    def gen_cb_url(self, i: int) -> str:
        return f"{BASE_URL}/shop/{self.value}?currentPage={i}"


def get_category_from_url(url: str) -> Category:
    return Category(url.split("/")[-1].split("?")[0])


def gen_reviews_url(offset: int, product_id: str):
    return f"https://api.bazaarvoice.com/data/reviews.json?Filter=contentlocale%3Aen*&Filter=ProductId%3A{product_id}&Limit=100&Offset={offset}&Include=Products%2CComments&Stats=Reviews&passkey=caQ0pQXZTqFVYA1yYnnJ9emgUiW59DXA85Kxry8Ma02HE&apiversion=5.4&Locale=en_US"
    
class ProductbotSpider(Spider):
    name = "productbot"
    start_urls = [
        category.gen_cb_url(i)
        for i in range(1, NUM_PAGES)
        for category in list(Category)
    ]

    def parse(self, response):
        i = 1
        product_url = self.get_product(response, i)
        while product_url:
            yield Request(
                url=product_url,
                callback=self.parse_ingredients,
                cb_kwargs=dict(category=get_category_from_url(response.url)),
                meta={"splash": {"endpoint": "render.html", "args": {"wait": 0.5}, "splash_headers": {"Connection": "keep-alive"}}},
            )
            i += 1
            product_url = self.get_product(response, i)

    def parse_ingredients(self, response, category: Category):
        item_data = json.loads(response.xpath("//*[@id='linkStore']/text()").get())
        ingredients = item_data[
            "page"
        ]["product"]["currentSku"]["ingredientDesc"].split("<br><br>")[1].split("<br>")[-1]
        brand = item_data["page"]["product"]["productDetails"]["brand"]["displayName"]
        product_name = item_data["page"]["product"]["productDetails"]["displayName"]
        avg_rating = item_data["page"]["product"]["productDetails"]["rating"]
        num_reviews = item_data["page"]["product"]["productDetails"]["reviews"]
        num_loves = item_data["page"]["product"]["productDetails"]["lovesCount"]

        # Request reviews
        offset = 0
        product_id = item_data["page"]["product"]["productDetails"]["productId"]
        reviews_url = gen_reviews_url(offset, product_id)
        review_data = []
        while True:
            reviews = requests.get(reviews_url).json()
            if reviews["Results"] == []:
                break
            review_data += reviews["Results"]
            offset += 100
            reviews_url = gen_reviews_url(offset, product_id)

        return {
            "ingredients": ingredients,
            "brand": brand,
            "product_name": product_name,
            "url": response.url,
            "category": category.name,
            "avg_rating": avg_rating,
            "num_reviews": num_reviews,
            "num_loves": num_loves,
            "review_data": review_data
        }

    def get_product(self, response, i: int) -> str:
        return response.xpath(
            f"//div[@class='css-1322gsb']/div[{i + 1}]//a[1]/@href"
        ).get()
