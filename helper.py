import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import unidecode
from truecaser.Truecaser import *
import truecaser.PredictTruecaser as truecase
from difflib import SequenceMatcher 

stop_words = set(stopwords.words('english'))	
forbidden_inclusion = ['mac', 'us', 'john']

def get_titles(gazetteer_dir):
	list_titles = set()
	with open(gazetteer_dir+'/jobtitles.lst', 'r') as gfile:
		for line in gfile.readlines():
			list_titles.add(line.strip().lower())
	with open(gazetteer_dir+'/title_lower.lst', 'r') as gfile:
		for line in gfile.readlines():
			list_titles.add(line.strip().lower())
	return list_titles

def get_names(text):
	names = []
	for name in re.findall(' ([A-Z][a-z][\w-]+(\s+[A-Z][a-z][\w-]+)*)', text):
		fi_name = ' '.join([w for w in word_tokenize(name[0]) if not w.lower() in stop_words])
		if fi_name != '':
			names.append(fi_name)
	return names

def get_speaker_names(text):
	names = []
	for name in re.split(r"[\s]{3,}|and|AND|'s|'S|[,&/]", text):
		fi_name = ' '.join([w for w in word_tokenize(name.strip()) if not w.lower() in stop_words])
		if fi_name != '' and len(fi_name) > 2:
			names.append(fi_name)
	return names

def get_clean_text(text):
	text = text.replace('-LRB-', '(')
	text = text.replace('-RRB-', ')')
	text = text.replace('`', '')
	text = text.replace('"', '')
	text = text.replace("''", '')
	text = re.sub(' +', ' ', text)
	#text = re.sub(' - ', '-', text)
	text = re.sub(r'([A-Z]{4,})\s([A-Z][a-z]+)', r'\1%s\2' % ' , ', text)
	text = unidecode.unidecode(text)
	return text

def get_cased_text(text):
	try:
		text_cased = truecase.get_true_case(text)
		text_cased = re.sub(r'\b%s\b' % "Tows", "tows", text_cased)
		return text_cased
	except KeyError:
		return text	

def longestSubstring(str1,str2): 
	# initialize SequenceMatcher object with  
	# input string 
	seqMatch = SequenceMatcher(None,str1,str2) 

	# find match of longest sub-string 
	# output will be like Match(a=0, b=0, size=5) 
	match = seqMatch.find_longest_match(0, len(str1), 0, len(str2)) 

	# print longest substring 
	if (match.size!=0): 
	  return (str1[match.a: match.a + match.size])  
	else: 
	  return None

def common_noun(w):
	return wn.synsets(w) != []

def check_eligible_name(w):
	if len(w) == 1:
		return not common_noun(w)
	else:
		all_common = True
		for ww in w.split(' '):
			all_common &= common_noun(ww)
		return not all_common

