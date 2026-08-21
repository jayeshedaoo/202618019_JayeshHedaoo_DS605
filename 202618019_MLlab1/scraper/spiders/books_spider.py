import scrapy


class BooksSpider(scrapy.Spider):
    name = "books"
    allowed_domains = ["books.toscrape.com"]

    start_urls = [
        "https://books.toscrape.com/catalogue/page-1.html"
    ]

    def parse(self, response):
        # Get all book links on the current page
        books = response.css("article.product_pod h3 a")

        for book in books:
            relative_url = book.attrib.get("href")

            yield response.follow(
                response.urljoin(relative_url),
                callback=self.parse_book
            )

        # Follow the next page
        next_page = response.css("li.next a::attr(href)").get()

        if next_page:
            yield response.follow(
                response.urljoin(next_page),
                callback=self.parse
            )

    def parse_book(self, response):
        title = response.css("div.product_main h1::text").get()

        price = response.css("p.price_color::text").get()

        rating = response.css(
            "p.star-rating::attr(class)"
        ).get(default="").replace("star-rating ", "")

        availability = response.css(
            "p.instock.availability::text"
        ).getall()

        availability = " ".join(
            [text.strip() for text in availability if text.strip()]
        )

        description = response.css(
            "#product_description + p::text"
        ).get(default="")

        breadcrumb = response.css(
            "ul.breadcrumb li a::text"
        ).getall()

        category = breadcrumb[-1] if breadcrumb else ""

        product_info = {}

        rows = response.css("table.table.table-striped tr")

        for row in rows:
            key = row.css("th::text").get()
            value = row.css("td::text").get()
            product_info[key] = value

        yield {
            "title": title,
            "category": category,
            "price": price,
            "rating": rating,
            "availability": availability,
            "product_description": description,
            "UPC": product_info.get("UPC"),
            "number_of_reviews": product_info.get("Number of reviews"),
            "product_url": response.url
        }