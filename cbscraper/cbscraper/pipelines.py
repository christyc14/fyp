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
        if (
            not item["ingredients"]
            or item["url"] in self.unique
            or "kit" in item["name"].lower()
            or "set" in item["name"].lower()
            or "system" in item["name"].lower()
            or any([item["name"].startswith(name) for name in BLACKLIST_NAMES])
        ):
            raise DropItem
        item["ingredients"] = list(map(str.strip, item["ingredients"].split(", ")))
        self.unique.add(item["url"])
        return item


BLACKLIST_NAMES = [
    "Sunday Riley A+ High-Dose Retinoid Serum",
    "Dr.Jart+ Cryo Rubber Mask",
    "NIOD Copper Amino Isolate Serum",
    "MALIN + GOETZ Salicylic Gel",
    "Pai Skincare",
    "Augustinus Bader Discovery Size The Cream",
    "Augustinus Bader The Cream",
    "La Mer The Oil Absorbing Tonic",
    "Chantecaille Rice & Geranium Foaming Cleanser",
    "Anthony SPF30 Day Cream",
    "Sunday Riley C.E.O. Vitamin C Rich Hydration Cream",
    "Alpha-H H8 3 Step",
    "Tan-Luxe Super Glow Intensive Night Treatment Mask",
    "Goldfaden MD Doctor's Scrub and Bright Eyes Duo",
    "Slip Silk Sleep Mask",
    "La Mer The Mist",
    "Shiseido Exclusive Vital Perfection LiftDefine Radiance Face Mask",
    "Chantecaille Flower Harmonizing Cream",
    "Sunday Riley A High-Dose Retinoid Serum",
    "FOREO",
    "Zitsticka Blur Potion Discoloration Brightening Supplement",
    "The Konjac Sponge Company",
    "Sunday Riley C.E.O. 15% Vitamin C Brightening Serum",
    "D\u00e9esse Pro D\u00e9esse Professional LED Mask Next Generation",
    "Chantecaille Bio Lifting Mask",
    "By Terry Cellularose CC Serum 30ml",
    "Chantecaille Gold Recovery Mask",
    "La Mer",
    "Chantecaille Flower Infused Cleansing Milk",
    "Furtuna Skin X Jeremy Scott Collector's Edition",
    "Dr. Barbara Sturm Brightening Face Cream"
]
