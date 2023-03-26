import requests
from bs4 import BeautifulSoup

url='https://www.bizhizu.cn/manhua/katong/'
try:    
    bs=BeautifulSoup(requests.get(url).content,'lxml')
    pic_id=0
except Exception as ex:
    print('ERROR')

src=bs.find('div',class_='zt_list_left').find_all('img')
try:
    for img in src:
        pic_url=img['src']
        print(pic_id,pic_url)
        with open('./pic_%s.jpg'%str(pic_id),'wb') as pic:
            pic.write(requests.get(pic_url).content)
            pic_id+=1
except Exception as ex:
    print("ERROR")

print('爬取完毕！')