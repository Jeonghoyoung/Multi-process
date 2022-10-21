from selenium import webdriver
from selenium.webdriver.common.by import By
from util.time_utils import clock
from tqdm import tqdm
import time
import random
import chromedriver_autoinstaller


def translate_crawl(site_info:dict, text_list):
    chrome_filename = chromedriver_autoinstaller.utils.get_chromedriver_filename()
    chrome_ver = chromedriver_autoinstaller.get_chrome_version().split('.')[0]
    options = webdriver.ChromeOptions()
    try:
        driver = webdriver.Chrome(f'./{chrome_ver}/{chrome_filename}', chrome_options=options)
    except:

        chromedriver_autoinstaller.install(True)

        driver = webdriver.Chrome(f'./{chrome_ver}/{chrome_filename}', chrome_options=options)
    driver.implicitly_wait(3)

    driver.get(site_info['url'])

    selected_tag_a = driver.find_element(By.CSS_SELECTOR, site_info['src_text'])
    time_rand = random.randint(3, 5)

    trans_text = [text + '\n' for text in text_list]
    trans_result = []

    for i in tqdm(range(0, len(trans_text), 50)):
        sublist = ''.join(trans_text[i:i + 50])

        selected_tag_a.send_keys(sublist)
        time.sleep(2)
        result = driver.find_element(By.CSS_SELECTOR,site_info['tgt_text']).text
        trans_result.append(result)
        time.sleep(1)
        selected_tag_a.clear()
    driver.quit()
    return trans_result

@clock
def papago_trans(words, src_lang, tgt_lang):
    info_dict = {
        'url': f'https://papago.naver.com/?sk={src_lang}&tk={tgt_lang}&hn=0',
        'src_text': 'textarea#txtSource',
        'tgt_text': '#txtTarget'
    }
    return translate_crawl(info_dict, words)

@clock
def google_trans(words, src_lang, tgt_lang):
    info_dict = {
        'url': f'https://translate.google.co.kr/?hl=ko&sl={src_lang}&tl={tgt_lang}&op=translate',
        'src_text': 'textarea.er8xn',
        'tgt_text': 'span.ryNqvb'
    }
    return translate_crawl(info_dict, words)


if __name__ == '__main__':
    text_list = ['안녕',
                 '반갑습니다.',
                 '너의 이름은']

    a = google_trans(text_list, 'ko', 'en')
    print(a)
