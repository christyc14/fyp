from ast import Dict
from enum import Enum, auto
import json
import re
from scrapy import Request, Spider

BASE_URL = "https://www.cultbeauty.com"
NUM_PAGES = 100
EMPTY_DESCRIPTION = "\n\n                              "


class Category(Enum):
    CLEANSER = "cleansers-toners.list"
    TONER = "toners-mists/toner.list"
    MOISTURISER = "moisturisers.list"
    SERUM = "serums.list"
    MASKS = "masks-exfoliators/masks.list"
    EXFOLIATORS = "masks-exfoliators/exfoliators.list"
    SPF = "moisturisers/spfs.list"

    def gen_cb_url(self, i: int) -> str:
        return f"{BASE_URL}/skin-care/{self.value}?pageNumber={i}"


def get_category_from_url(url: str) -> Category:
    return Category(url.split("/", 4)[-1].split("?")[0])


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
                url=BASE_URL + product_url,
                callback=self.parse_ingredients,
                cb_kwargs=dict(category=get_category_from_url(response.url)),
            )
            i += 1
            product_url = self.get_product(response, i)

    def parse_ingredients(self, response, category: Category):
        ingredients = response.xpath(
            '//div[@id="product-description-content-7"]//p/text()'
        ).get()
        if ingredients == EMPTY_DESCRIPTION:
            ingredients = response.xpath(
                '//div[@id="product-description-content-7"]//div/text()'
            ).get()
        return {
            "ingredients": ingredients,
            "name": response.xpath("//h1[1]/text()").get(),
            "url": response.url,
            "category": category.name,
        }

    def get_product(self, response, i: int) -> str:
        return response.xpath(
            f"//ul[@class='productListProducts_products']/li[{i}]//a[1]/@href"
        ).get()
