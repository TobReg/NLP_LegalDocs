#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 22:07:53 2020

@author: toby
"""


import scrape_schema_recipe
import rdflib
import json

from rdflib import Graph, plugin
import json, rdflib_jsonld
from rdflib.plugin import register, Serializer
register('json-ld', Serializer, 'rdflib_jsonld.serializer', 'JsonLDSerializer')


url = 'https://www.foodnetwork.com/recipes/alton-brown/honey-mustard-dressing-recipe-1939031'
url = 'https://foodnetwork.co.uk/recipes/honey-mustard-dressing/?utm_source=foodnetwork.com&utm_medium=domestic'
url = 'https://www.chefkoch.de/rezepte/2569781402262652/Buchweizen-mit-Pilzen.html'
url = 'https://www.allrecipes.com/recipe/246628/spaghetti-cacio-e-pepe/'
url = 'https://dieseekocht.com/2019/04/30/pasta-con-le-sarde-original-rezept-aus-sizilien/'
url = 'https://sz-magazin.sueddeutsche.de/das-rezept/kuerbis-mangold-quiche-mit-speck-und-apfel-89284/'
url = 'https://www.brigitte.de/rezepte/herbstkueche--pilzrezepte--wir-feiern-den-herbst--10651814.html'
url = 'https://www.epicurious.com/recipes/food/views/arepas-51245240'


recipe_list = scrape_schema_recipe.scrape_url(url, python_objects=True)
len(recipe_list)


recipe = recipe_list[0]
#test = recipe_list[1]
type(recipe)

len(recipe)

for key in recipe:
	print (key)

# Name of the recipe
recipe['name']

# List of the Ingredients
recipe['recipeIngredient']

# List of the Instructions
recipe['recipeInstructions']

# Author
recipe['author']

recipe['aggregateRating']
recipe['aggregateRating']['ratingValue']

############### scraping a web site to collect its subpages ###########

from bs4 import BeautifulSoup 
import requests 
import requests.exceptions 
from urllib.parse import urlsplit 
from urllib.parse import urlparse 
from collections import deque

url = "https://scrapethissite.com"
url = "https://tobyregner.info"
#url = 'https://www.chefkoch.de/rezepte/'
url = "https://www.epicurious.com/search/?special-consideration=low-no-sugar%2Cwheat-gluten-free&cuisine=italian"

# a queue of urls to be crawled next
new_urls = deque([url])

# a set of urls that we have already processed 
processed_urls = set()

# a set of domains inside the target website
local_urls = set()

# a set of domains outside the target website
foreign_urls = set()

# a set of broken urls
broken_urls = set()

# process urls one by one until we exhaust the queue
while len(new_urls):    # move url from the queue to processed url set
    url = new_urls.popleft()
    processed_urls.add(url) 
    # print the current url
    print("Processing %s" % url)

    response = requests.get(url)
#    try:
#        response = requests.get(url)
#    except(requests.exceptions.MissingSchema, requests.exceptions.ConnectionError, requests.exceptions.InvalidURL, requests.exceptions.InvalidSchema):    # add broken urls to it’s own set, then continue    
#        broken_urls.add(url)
#        continue

# extract base url to resolve relative links
    parts = urlsplit(url)
    base = "{0.netloc}".format(parts)
    strip_base = base.replace("www.", "")
    base_url = "{0.scheme}://{0.netloc}".format(parts)
    path = url[:url.rfind('/')+1] if '/' in parts.path else url

    soup = BeautifulSoup(response.text, "lxml")

    for link in soup.find_all('a'):    # extract link url from the anchor
        anchor = link.attrs["href"] if "href" in link.attrs else ''
        if anchor.startswith('/'):
            local_link = base_url + anchor
            local_urls.add(local_link)
        elif strip_base in anchor:
            local_urls.add(anchor)
        elif not anchor.startswith('http'):
            local_link = path + anchor
            local_urls.add(local_link)
        else:
            foreign_urls.add(anchor)
    
    for i in local_urls:
        if not i in new_urls and not i in processed_urls:
            new_urls.append(i) 
            


len(local_urls)

recipe_urls = list(local_urls)

#recipe_urls = [i.split('/')[3] for i in recipe_urls]


import pandas as pd

recipe_urls = pd.DataFrame(recipe_urls)
recipe_urls.columns = ['URL']

recipe_urls['check'] = recipe_urls['URL'].str.split('/').str[3]

recipe_urls['check'].value_counts()

recipe_urls = recipe_urls[recipe_urls['check'] == 'recipes']

url_input = recipe_urls['URL']

url_input = url_input[4:14]
url_input.str.split('/').str[6]
len(url_input)

dict_recipes = dict()
y = 1

for x in url_input:
    recipe_list = scrape_schema_recipe.scrape_url(x, python_objects=True) 
    dict_recipes[y] = recipe_list[0]
    y = y + 1



dict_recipes[2]['name']

          