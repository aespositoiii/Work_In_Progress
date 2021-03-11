from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time

driver = webdriver.Safari()
time.sleep(5)
driver.get("https://www.mouser.com/")
time.sleep(5)
driver.find_element_by_css_selector('#lnkViewAllProductCategories').click()
'''time.sleep(10)
assert "Python" in driver.title
time.sleep(10)
elem = driver.find_element_by_name("q")
time.sleep(10)
elem.clear()
time.sleep(10)
elem.send_keys("pycon")
time.sleep(10)
elem.send_keys(Keys.RETURN)
time.sleep(10)
assert "No results found." not in driver.page_source'''
time.sleep(5)
driver.close()