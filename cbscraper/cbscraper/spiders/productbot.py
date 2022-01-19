import json
import scrapy

BASE_URL = "https://www.cultbeauty.com"
NUM_PAGES = 100

def gen_page_url(i: int) -> str:
    return f'https://www.cultbeauty.com/skin-care.list?pageNumber={i}'

class ProductbotSpider(scrapy.Spider):
    name = 'productbot'
    start_urls = [gen_page_url(i) for i in range(1, NUM_PAGES)]

    def parse(self, response):
        i = 1
        product_url = self.get_product(response, i)
        while product_url:
            yield {"url": product_url}
            i += 1
            product_url = self.get_product(response, i)

    def get_product(self, response, i: int) -> str:
        return response.xpath(f"//ul[@class='productListProducts_products']/li[{i}]//a[1]/@href").get()
