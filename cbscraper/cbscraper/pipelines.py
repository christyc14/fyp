# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
from scrapy.exceptions import DropItem


class CbscraperPipeline:
    unique = set()

    def process_item(self, item, spider):
        if not item["ingredients"] or item["url"] in self.unique or "kit" in item["name"].lower():
            raise DropItem
        item["ingredients"] = list(map(str.strip, item["ingredients"].split(", ")))
        self.unique.add(item["url"])
        return item
