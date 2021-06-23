import requests
from bs4 import BeautifulSoup
from urllib.request import urlopen
from bs4 import BeautifulSoup

html = urlopen("https://www.weather.go.kr/plus/land/current/aws_table_popup.jsp")  
soup = BeautifulSoup(html, "lxml") 

soup
table = soup.find("table",class_="forecastNew3")    #웹크롤링을 시작하는 태그
