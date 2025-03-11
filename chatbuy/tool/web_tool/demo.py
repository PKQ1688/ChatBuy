import asyncio

from pydoll.browser.chrome import Chrome
from pydoll.constants import By


async def main():
    """Main function to start the browser, navigate to a page, and click a button."""
    # Start the browser with no additional webdriver configuration!
    async with Chrome() as browser:
        await browser.start()
        page = await browser.get_page()
        # Navigate through captcha-protected sites without worry
        await page.go_to("https://github.com/thalissonvs/pydoll")
        button = await page.find_element(By.CSS_SELECTOR, "button")
        await button.click()


asyncio.run(main())
