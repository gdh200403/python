import requests

url='http://vi.ustc.edu.cn/_upload/article/images/e5/4f/b1f558a04652bee07d23d4ce7e4c/W020110714574903022716.jpg'
print('-'*30+'download image %s'%url)
try:
    response=requests.get(url)
    print(response.status_code)
    image=response.content
    with open('./学习爬虫/ustc_img.jpg','wb') as f:
        f.write(image)
except Exception as ex:
    print('----------ERROR----------',ex)
