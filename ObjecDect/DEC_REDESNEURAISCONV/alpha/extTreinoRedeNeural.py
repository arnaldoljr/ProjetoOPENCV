import scrapy
#from scrapy.contrib.loader import ItemLoader
#import ItemLoader
    
class YoutubeVideo(scrapy.Item):
    link = scrapy.Field()
    title = scrapy.Field()
    views = scrapy.Field()
 
class YoutubeChannelLister(scrapy.Spider):
    name = 'youtube-lister'
    #youtube_channel = 'LongboardUK'
    #start_urls = ['https://www.youtube.com/user/%s/videos' % youtube_channel]
    start_url = ['https://www.youtube.com/results?search_query=roubo+assalto+furto+cameras+de+seguran%C3%A7a+']
    def parse(self, response):
        i = 0 
        while( i <301):
            for sel in response.css("ul#channels-browse-content-grid > li"):
                loader = ItemLoader(YoutubeVideo(), selector=sel)

                loader.add_xpath('link', './/h3/a/@href')
                loader.add_xpath('title', './/h3/a/text()')
                loader.add_xpath('views', ".//ul/li[1]/text()")

                yield loader.load_item()
                i=i+1
       
