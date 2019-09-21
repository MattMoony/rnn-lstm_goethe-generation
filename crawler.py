import os
import re
import selenium
from tqdm import tqdm
import requests as req
from bs4 import BeautifulSoup
from selenium import webdriver
from argparse import ArgumentParser

# data-source:  http://www.zeno.org/Literatur/M/Goethe,+Johann+Wolfgang/Gedichte 
#               (http://www.zeno.org/Zeno/-/Lizenz%3A+Gemeinfrei)

def grab_sub_points(browser):
    return browser.find_elements_by_class_name('zenoPLm8n8')

def main():
    parser = ArgumentParser()

    parser.add_argument('-d', '--destination', type=str, dest='dest', default='data/', 
                        help='Path to the destination folder')
    parser.add_argument('-v', '--verbose', action='store_true', dest='verbose', 
                        help='Show browser window?')
    
    args = parser.parse_args()

    if not os.path.isdir(args.dest):
        try:
            os.mkdir(args.dest)
        except Exception:
            print('[-] Couldn\'t create destination directory ... ')
            os._exit(1)

    options = webdriver.FirefoxOptions()
    options.headless = not args.verbose

    browser = webdriver.Firefox(firefox_options=options)

    browser.set_window_position(25, 25)
    browser.set_window_size(1000, 800)

    browser.get('http://www.zeno.org/Literatur/M/Goethe,+Johann+Wolfgang/Gedichte')
    sub_points  = grab_sub_points(browser)
    sub_hrefs   = []

    for sp in tqdm(sub_points, desc='Fetching URLs ...    ', unit='url'):
        try:
            a       = sp.find_element_by_class_name('zenoTXLinkInt')
            href    = a.get_attribute('href')

            sub_hrefs.append(href)
        except selenium.common.exceptions.NoSuchElementException:
            pass

    for href in tqdm(sub_hrefs, desc='Processing poems ... ', unit='poe'):
        res = req.get(href)
        try:
            soup = BeautifulSoup(res.text, 'lxml')
        except Exception:
            continue

        ma = soup.find('div', class_='zenoCOMain')
        try:
            title = ma.find('h4').text
        except Exception:
            continue
        content = []

        for e in ma.find_all(['p', 'br']):
            if e.name == 'p':
                content.append(e.text)
            else:
                content.append('')

        fname   = re.sub(r'\s', '_', title.lower())
        fname   = re.sub(r'[^\w\d]', '', fname)
        fpath   = os.path.join(args.dest, fname + '.txt')
        index   = -1

        while os.path.isfile(fpath):
            index += 1
            fpath = os.path.join(args.dest, '{}-{}.txt'.format(fname, index))

        with open(fpath, 'wb') as f:
            f.write(title.encode('utf-8'))
            f.write('\n\n\n'.encode('utf-8'))
            f.write('\n'.join(content).encode('utf-8'))
            f.write('\n\n'.encode('utf-8'))

    browser.quit()

if __name__ == '__main__':
    main()