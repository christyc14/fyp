# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
import re
from itemadapter import ItemAdapter
from scrapy.exceptions import DropItem


class CbscraperPipeline:
    unique = set()

    def process_item(self, item, spider):
        if (
            not item["ingredients"]
            or item["url"] in self.unique
            or "kit" in item["product_name"].lower()
            or "set" in item["product_name"].lower()
            or "system" in item["product_name"].lower()
            or any([item["product_name"].startswith(name) for name in BLACKLIST_NAMES])
        ):
            raise DropItem
        item["ingredients"] = list(map(str.strip, re.split(",(?![^()]*\))", item["ingredients"])))
        self.unique.add(item["url"])
        return item


BLACKLIST_NAMES = [
    "Cr\u00e8me Ancienne\u00ae Soft Cream",
    "Repairwear Uplifting Firming Cream Broad Spectrum SPF 15",
    "GinZing\u2122 Oil- Free Energy Boosting Gel Moisturizer",
    "Clean Screen Mineral SPF 30 Mattifying Face Sunscreen",
    "Superkind Fragrance-Free Fortifying Moisturizer",
    "Perfectly Clean Multi-Action Toning Lotion/Refiner",
    "Black Label Detox BB Beauty Balm SPF 30",
    "Turnaround Overnight Revitalizing Moisturizer",
    "Even Better Skin Tone Correcting Moisturizer Broad Spectrum SPF 20",
    "Repairwear Laser Focus Line Smoothing Cream Broad Spectrum SPF 15 for Very Dry to Dry Combination Skin",
    "Revitalizing Supreme+ Night Intensive Restorative Cr\u00e8me Moisturizer",
    "Vitamin C+ Collagen Deep Cream",
    "The Radiant SkinTint SPF 30",
    "Triple Vitamin C Brightening Bounce Cream Moisturizer",
    "Seaberry Moisturizing Face Oil",
    "Pore Refining Solutions Stay-Matte Hydrator",
    "Mini The Moisturizing Cool Gel Cream",
    "Extra Illuminating Moisture Balm",
    "The Tonic"]
