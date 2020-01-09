import scrapy
import urllib.parse
 
class Noticia(scrapy.Item):
    titulo = scrapy.Field()
    conteudo = scrapy.Field()
    link = scrapy.Field()
    data_publicacao = scrapy.Field()
 
class PortalBrasilDestaques(scrapy.Spider):
    name = 'portal-brasil'
 
    def __init__(self, assunto=None):
        main_url = 'http://www.brasil.gov.br'
        if assunto:
            self.start_urls = ['%s/%s' % (main_url, assunto)]
        else:
            self.start_urls = [main_url]
 
    def parse(self, response):
        """Recebe a pagina com as noticias destaques, encontra os links
        das noticias e gera requisicoes para a pagina de cada uma
        """
        links_noticias = response.xpath(
            "//div/h1/a/@href"
            " | //div/h3/a/@href[not(contains(.,'conteudos-externos'))]"
        ).extract()
 
        for link in links_noticias:
            url_noticia = urlparse.urljoin(response.url, link)
            yield scrapy.Request(url_noticia, self.extrai_noticia)
 
    def extrai_noticia(self, response):
        """Recebe a resposta da pagina da noticia,
        e extrai um item com a noticia
        """
        noticia = Noticia()
 
        noticia['link'] = response.url
        noticia['titulo'] = response.xpath('&amp;amp;amp;quot;//article/h1/text()&amp;amp;amp;quot;')[0].extract()
        noticia['conteudo'] = response.xpath(
            "string(//div[@property='rnews:articleBody'])")[0].extract()
        noticia['data_publicacao'] = ''.join(
            response.css('span.documentPublished::text').extract()).strip()
 
        yield noticia
