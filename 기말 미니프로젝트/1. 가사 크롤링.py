# 가사 크롤링 코드
# 1000곡 크롤링

from selenium.webdriver.common.by import By
import re

pages = [] # 크롤링 오류 확인 위해 page 수 기록
title = []
lyric = []
artist = []

p = re.compile("'(\d+)'")
for page in range(1, 1001, 50):  # 1페이지만 추출(1,51,50), 발라드 gnrCode=GN0100, 댄스 gnrCode=GN0200
    url = "https://www.melon.com/genre/song_list.htm?gnrCode=GN0100&steadyYn=Y".format(page)  # 댄스 페이지
    driver.get(url)

    time.sleep(2)  # 각자의 컴퓨터 성능에 따라 설정

    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')

    # 한 페이지에 있는 50개의 songId 추출
    songid = []  # songid

    for i in range(50):
        data = soup.find_all("a", class_="btn button_icons type03 song_info")[i].get('href')
        id_ = p.search(str(data)).group(1)
        songid.append(id_)

    # 해당 songId 입력 후 정보 추출
    for song in songid:
        lyric_url = 'https://www.melon.com/song/detail.htm?songId={}'.format(song)
        driver.get(lyric_url)

        time.sleep(2)

        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')

        a = soup.select('div.song_name')[0].text[3:].strip()  # 제목
        title.append(a)

        try:
            driver.find_element(By.CSS_SELECTOR, ".button_more.arrow_d").click()  # 가사
            ly = driver.find_element(By.CSS_SELECTOR, 'div#d_video_summary')
            b = ly.text
            lyric.append(b)
        except:
            lyric.append('가사없음')
        ar = driver.find_elements(By.CSS_SELECTOR, "div.artist")
        for value in ar:
            c = value.text
        artist.append(c)

        pages.append(page)

driver.quit()
