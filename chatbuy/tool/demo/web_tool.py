import asyncio

from pydoll.browser.chrome import Chrome
from pydoll.constants import By


async def main():
    """Main function to start the browser and navigate to a page."""
    # Start the browser with no additional webdriver configuration!
    async with Chrome() as browser:
        await browser.start()
        page = await browser.get_page()

        # Navigate through captcha-protected sites without worry
        await page.go_to("https://github.com/thalissonvs/pydoll")
        button = await page.find_element(By.CSS_SELECTOR, "#repo-content-pjax-container > div > div > div > div.Layout-main > react-partial > div > div > div.Box-sc-g0xbh4-0.vIPPs > div.Box-sc-g0xbh4-0.csrIcr > div > div.Box-sc-g0xbh4-0.QkQOb.js-snippet-clipboard-copy-unpositioned.undefined > article > ul:nth-child(13) > li:nth-child(1) > a")
        await button.click()


asyncio.run(main())
