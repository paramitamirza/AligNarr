import sys
import re
import xml.etree.ElementTree as ET
from lxml import etree
import os
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
import subprocess
import codecs
from nltk.corpus import wordnet as wn
import helper as pre
import name_linking as link
import wikipedia
import wikipediaapi
import csv
import pandas as pd

dir_path = os.path.dirname(os.path.realpath(__file__))
dataset_dir = dir_path + '/dataset/10_movies/'
output_dir = dir_path + '/output_files/'

movie_code = {'anastasia': 'Anastasia', 
				'cars': 'Cars_2', 
				'cast': 'Cast_Away', 
				'pulp': 'Pulp_Fiction', 
				'shrek': 'Shrek', 
				'south': 'South_Park', 
				'swordfish': 'Swordfish', 
				'butterfly': 'The_Butterfly_Effect', 
				'silence': 'The_Silence_of_the_Lambs', 
				'walle': 'WALL-E'
				}

stop_words = set(stopwords.words('english'))	

nlp = spacy.load("en_core_web_lg")

def get_wiki_plot(movie_title, filepath, load=True, wiki_title=''):
	sum_sentences = ''
	cast_sentences = ''
	if os.path.isfile(filepath) and load:
		with open(filepath, encoding='utf-8') as f:
			sum_sents = []
			for line in f.readlines():
				if line.strip() != "":
					sent = ' '.join(word_tokenize(line.strip()))
					sum_sents.append(sent)

			sum_sentences = '\n'.join(sum_sents)

	else:
		#get plot from wikipedia
		wiki_wiki = wikipediaapi.Wikipedia(language='en', extract_format=wikipediaapi.ExtractFormat.WIKI)

		if wiki_title == '':
			movie_page = wikipedia.search(movie_title.replace('_', ' '))[0]
			movie = wiki_wiki.page(movie_page)
		else:
			movie = wiki_wiki.page(wiki_title)
		for s in movie.sections:
			if s.title.startswith('Plot') or s.title.startswith('Synopsis'):
				text = ''
				if len(s.sections) > 0:
					for ss in s.sections:
						text += ss.text
				else:
					text = s.text
				doc = nlp(text)
				sum_sents = [sent.string.strip() for sent in doc.sents]
				sum_sentences = '\n'.join(sum_sents)

			elif s.title.startswith('Cast'):
				text = ''
				if len(s.sections) > 0:
					for ss in s.sections:
						text += ss.text
				else:
					text = s.text
				doc = nlp(text)
				cast_sents = [sent.string.strip() for sent in doc.sents if (sent.string.strip().endswith('.') and sent.string.strip().count(' ') > 5)]
				cast_sentences = '\n'.join(cast_sents)

	return (sum_sentences, cast_sentences)

def process_script_and_summary(script_xml, wikiplot_sentences):
	tree = ET.parse(script_xml)
	root = tree.getroot()

	script_names = dict()
	summary_names = dict()
	names_original = dict()
	script_sentences = []
	summary_sentences = []
	stage_sentences = []

	num_scenes = 0
	num_summary_sentences = 0

	### script scenes ###
	for scene in root.iter('scene'):
				
		stageText = ''
		stage = scene.find('stageDirection')
		if stage:
			stageText = stage.text
			if stageText:
				stageText = re.sub(' +', ' ', stageText)
				stageText = pre.get_clean_text(stageText)
				stageText = pre.get_cased_text(stageText)

				for name in pre.get_names(stageText):
					name_as_key = name.lower()
					if name_as_key not in script_names:
						next_idx = len(script_names)
						script_names[name_as_key] = "E"+str(next_idx)
					if name_as_key not in names_original:
						names_original[name_as_key] = set()
					names_original[name_as_key].add(name)

		stage_sentences.append('s'+scene.get('count')+' ### '+stageText)

		sceneText = ''	
		for child in scene.getchildren():
			if child.tag == 'description':		
				#print('desc-', child.get('count'))
				descText = ''
				for sentence in child.iter('sentence'):
					sent = ''
					for word in sentence.iter('word'):
						sent += word.text + ' '

					sent_clean = pre.get_clean_text(sent)

					for name in pre.get_names(pre.get_cased_text(sent_clean)):
						name_as_key = name.lower()
						if name_as_key not in script_names:
							next_idx = len(script_names)
							script_names[name_as_key] = "E"+str(next_idx)
						if name_as_key not in names_original:
							names_original[name_as_key] = set()
						names_original[name_as_key].add(name)
					
					script_sentences.append('s'+scene.get('count')+'d'+child.get('count')+' ### '+sent_clean)

					descText += sent_clean + '#NEWLINE#'
				
				sceneText += descText.replace("#NEWLINE#", "<br/>")
			
			elif child.tag == 'speech':
				descText = ''
				for sentence in child.iter('sentence'):
					sent = ''
					for word in sentence.iter('word'):
						sent += word.text + ' '
					
					sent_clean = pre.get_clean_text(sent)				

					for name in pre.get_names(sent_clean):
						name_as_key = name.lower()
						if name_as_key not in script_names:
							next_idx = len(script_names)
							script_names[name_as_key] = "E"+str(next_idx)
						if name_as_key not in names_original:
							names_original[name_as_key] = set()
						names_original[name_as_key].add(name)
					
					script_sentences.append('s'+scene.get('count')+'u'+child.get('count')+' ### '+child.get('speaker') + " ### " + sent_clean)

					descText += sent_clean + '#NEWLINE#'

				speaker_name = child.get('speaker')
				for name in pre.get_speaker_names(speaker_name):
					name_as_key = name.lower()
					if name_as_key not in script_names:
						next_idx = len(script_names)
						script_names[name_as_key] = "E"+str(next_idx)
					if name_as_key not in names_original:
						names_original[name_as_key] = set()
					names_original[name_as_key].add(name)
					names_original[name_as_key].add(re.sub('-', ' - ', name).strip())


				sceneText += "&nbsp;&nbsp;&nbsp;<strong>" + child.get('speaker') + "</strong>: <em>" + descText.replace("#NEWLINE#", " ") + "</em><br/>"

		num_scenes += 1

	### summary sentences ###
	idx = 0
	for sent in wikiplot_sentences.splitlines():
		doc = nlp(sent)
		tokens = []
		for token in doc:
			tokens.append(token.text)

		sent_clean = pre.get_clean_text(' '.join(tokens))				

		summary_sentences.append('s'+str(idx)+' ### '+sent_clean)
		num_summary_sentences += 1
		idx += 1

		for name in pre.get_names(sent_clean):
			name_as_key = name.lower()
			if name_as_key not in script_names and name_as_key not in summary_names:
				next_idx = len(script_names) + len(summary_names)
				summary_names[name_as_key] = "E"+str(next_idx)
			if name_as_key not in names_original:
				names_original[name_as_key] = set()
			names_original[name_as_key].add(name)

	return (script_names, summary_names, names_original, script_sentences, stage_sentences, summary_sentences, num_scenes, num_summary_sentences)

def process_summary(wikiplot_sentences, script_names, names_original):
	summary_names = dict()
	summary_sentences = []
	num_summary_sentences = 0

	### summary sentences ###
	idx = 0
	for sent in wikiplot_sentences.splitlines():
		doc = nlp(sent)
		tokens = []
		for token in doc:
			tokens.append(token.text)

		sent_clean = pre.get_clean_text(' '.join(tokens))				

		summary_sentences.append('s'+str(idx)+' ### '+sent_clean)
		num_summary_sentences += 1
		idx += 1

		for name in pre.get_names(sent_clean):
			name_as_key = name.lower()
			if name_as_key not in script_names and name_as_key not in summary_names:
				next_idx = len(script_names) + len(summary_names)
				summary_names[name_as_key] = "E"+str(next_idx)
			if name_as_key not in names_original:
				names_original[name_as_key] = set()
			names_original[name_as_key].add(name)

	return (summary_names, names_original, summary_sentences, num_summary_sentences)

def build_movie_xml(summary_replaced_html, stage_replaced_html, script_replaced_html):
	film = etree.Element("film")
	scenes_tree = etree.SubElement(film, "scenes")
	summary_tree = etree.SubElement(film, "summaries")

	for summary_sent in summary_replaced_html.splitlines():
		#print(summary_sent)
		idx = summary_sent.split(' ### ')[0].replace('s', '')
		sent = summary_sent.split(' ### ')[1]
		summary_node = etree.SubElement(summary_tree, 'summary', count=idx)
		summary_node.text = etree.CDATA(sent)	

	prev_scene = '0'
	curr_scene = ''
	prev_idx = ''
	curr_sent = ''
	for scene_sent in script_replaced_html.splitlines():
		split = scene_sent.split(' ### ')
		idx = split[0].replace('s', '')
		
		if len(split) > 2:
			speaker = split[1]
		sent = split[-1]

		if idx != prev_idx:

			if 'u' in prev_idx:
				curr_sent += "</em><br/>"

			if prev_idx != '':
				ids = re.split('d|u', prev_idx)[0]
				if ids != prev_scene:
					#print(ids, prev_idx, curr_scene[:10])
					scene_node = etree.SubElement(scenes_tree, 'scene', count=prev_scene, stage=stage_replaced_html.splitlines()[int(prev_scene)].split(' ### ')[1])
					scene_node.text = etree.CDATA(curr_scene)
					curr_scene = curr_sent
					prev_scene = ids
				else:
					curr_scene += curr_sent

			if 'u' in idx: 
				curr_sent = "&nbsp;&nbsp;&nbsp;<strong>" + speaker + '</strong> : <em>' + sent
			else:
				curr_sent = sent + "<br/>"
			
			prev_idx = idx

		else:
			if 'u' in idx: 
				curr_sent += ' ' + sent
			else:
				curr_sent += sent + ' <br/>'

	ids = re.split('d|u', prev_idx)[0]
	if 'd' in prev_idx:
		curr_scene += curr_sent
	else:
		curr_scene += curr_sent + "</em><br/>"
	scene_node = etree.SubElement(scenes_tree, 'scene', count=ids, stage=stage_replaced_html.splitlines()[int(prev_scene)].split(' ### ')[1])
	scene_node.text = etree.CDATA(curr_scene)

	return film

def write_movie_xml(code, summary_replaced_html, stage_replaced_html, script_replaced_html):
	film = build_movie_xml(summary_replaced_html, stage_replaced_html, script_replaced_html)

	if not os.path.exists(output_dir+'film_xml'): os.makedirs(output_dir+'film_xml/')
	with open(output_dir+'film_xml/' + code + '.xml', 'wb') as f:
		f.write(etree.tostring(film))

def write_movie_entities(code, entity_orig_names):
	if not os.path.exists(output_dir+'entities'): os.makedirs(output_dir+'entities/')
	with open(output_dir+'entities/'+code+'.json', 'w') as outfile:
		json.dump(entity_orig_names, outfile)

def write_init_alignment(code, num_scenes, num_summary_sentences):
	if not os.path.exists(output_dir+'alignment'): os.makedirs(output_dir+'alignment/')
	#if not os.path.exists('./output_files/alignment/'+code+'.json'):
	with open(output_dir+'alignment/'+code+'.json', 'w') as outfile:
		obj = {}
		if 'scene' not in obj: obj['scene'] = {}
		if 'summary' not in obj: obj['summary'] = {}    
		obj['num_scenes'] = num_scenes
		obj['num_summary_sentences'] = num_summary_sentences
		json.dump(obj, outfile)

def write_alignarr_files(dataset_dir, movie_title, summary_replaced_html, stage_replaced_html, script_replaced_html, entity_orig_names, num_scenes, num_summary_sentences):
	film = build_movie_xml(summary_replaced_html, stage_replaced_html, script_replaced_html)

	dir_path = os.path.join(dataset_dir, movie_title, '')
	if not os.path.exists(dir_path): os.makedirs(dir_path)
	with open(dir_path+'alignarr_script.xml', 'wb') as f:
		f.write(etree.tostring(film))
	with open(dir_path+'alignarr_entities.json', 'w') as outfile:
		json.dump(entity_orig_names, outfile)
	with open(dir_path+'alignarr_alignment.json', 'w') as outfile:
		obj = {}
		if 'scene' not in obj: obj['scene'] = {}
		if 'summary' not in obj: obj['summary'] = {}    
		obj['num_scenes'] = num_scenes
		obj['num_summary_sentences'] = num_summary_sentences
		json.dump(obj, outfile)

def write_resolved_summary(dataset_dir, movie_title, summary_replaced_sentences, entity_rep_names, filename='alignarr_summary.txt'):
	dir_path = os.path.join(dataset_dir, movie_title, '')
	if not os.path.exists(dir_path): os.makedirs(dir_path)
	with open(os.path.join(dir_path, filename), 'w') as f:
		summary_sentences = []
		for sent in summary_replaced_sentences:
			summary_sentences.append(sent.split(' ### ')[1].strip())

		summary_resolved_sentences = '\n'.join(summary_sentences)

		for k in entity_rep_names:
			summary_resolved_sentences = re.sub(r'\b%s\b' % k, entity_rep_names[k].title(), summary_resolved_sentences)

		f.write(summary_resolved_sentences)

def preprocess_movie(code, dataset_dir, coreference):
	script_xml = dataset_dir+'/' + movie_code[code] + '/script.xml'
	(wikiplot_sentences, _) = get_wiki_plot(movie_title, dataset_dir+'/' + movie_code[code] + '/wikiplot.txt')

	(script_names, summary_names, names_original, script_sentences, stage_sentences, summary_sentences, num_scenes, num_summary_sentences) = process_script_and_summary(script_xml, wikiplot_sentences)
	(script_names, summary_names, entity_names) = link.get_entities(script_names, summary_names)
	(entity_orig_names, entity_rep_names, summary_replaced_sentences, stage_replaced_sentences, script_replaced_sentences, summary_replaced_html, stage_replaced_html, script_replaced_html) = link.replace_linked_entities(entity_names, names_original, script_sentences, stage_sentences, summary_sentences, coreference=coreference)

	return (entity_orig_names, entity_rep_names, summary_replaced_sentences, stage_replaced_sentences, script_replaced_sentences, num_scenes, num_summary_sentences)

def run(code, dataset_dir, coreference=True):
	script_xml = dataset_dir+'/' + movie_code[code] + '/script.xml'
	movie_title = movie_code[code].replace('_', ' ')
	(wikiplot_sentences, _) = get_wiki_plot(movie_title, dataset_dir+'/' + movie_code[code] + '/wikiplot.txt')

	(script_names, summary_names, names_original, script_sentences, stage_sentences, summary_sentences, num_scenes, num_summary_sentences) = process_script_and_summary(script_xml, wikiplot_sentences)
	(script_names, summary_names, entity_names) = link.get_entities(script_names, summary_names)
	(entity_orig_names, entity_rep_names, summary_replaced_sentences, stage_replaced_sentences, script_replaced_sentences, summary_replaced_html, stage_replaced_html, script_replaced_html) = link.replace_linked_entities(entity_names, names_original, script_sentences, stage_sentences, summary_sentences, coreference=coreference)

	### Write files for annotation ###
	write_movie_xml(code, summary_replaced_html, stage_replaced_html, script_replaced_html)
	write_movie_entities(code, entity_orig_names)
	write_init_alignment(code, num_scenes, num_summary_sentences)

def run_scriptbase(dataset_dir, coreference=True):
	with open('scriptbase_statistic.csv', 'w', newline='') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(['movie', 'scenes', 'sentences', 'ratio'])

		for movie_title in os.listdir(dataset_dir):

			if os.path.exists(os.path.join(dataset_dir, movie_title, 'script.xml')):
				script_xml = os.path.join(dataset_dir, movie_title, 'script.xml')
				(wikiplot_sentences, _) = get_wiki_plot(movie_title, os.path.join(dataset_dir, movie_title, 'wikiplot.txt'))

				if script_xml and wikiplot_sentences:
					print('Processing', movie_title, '...')
					(script_names, summary_names, names_original, script_sentences, stage_sentences, summary_sentences, num_scenes, num_summary_sentences) = process_script_and_summary(script_xml, wikiplot_sentences)
					(script_names, summary_names, entity_names) = link.get_entities(script_names, summary_names)
					(entity_orig_names, entity_rep_names, summary_replaced_sentences, stage_replaced_sentences, script_replaced_sentences, summary_replaced_html, stage_replaced_html, script_replaced_html) = link.replace_linked_entities(entity_names, names_original, script_sentences, stage_sentences, summary_sentences, coreference=coreference)

					ratio = num_scenes / float(num_summary_sentences)
					writer.writerow([movie_title, num_scenes, num_summary_sentences, ratio])

					### Write files for annotation ###
					write_alignarr_files(dataset_dir, movie_title, summary_replaced_html, stage_replaced_html, script_replaced_html, entity_orig_names, num_scenes, num_summary_sentences)
					write_resolved_summary(dataset_dir, movie_title, summary_replaced_sentences, entity_rep_names)

				else:
					print('Processing', movie_title, 'failed.')

	df = pd.read_csv('scriptbase_statistic.csv')
	sorted_df = df.sort_values(by=["scenes"], ascending=True)
	sorted_df.to_csv('scriptbase_statistic_sorted.csv', index=False)
	
if __name__ == "__main__":
	if len(sys.argv) > 2 and sys.argv[1] == 'scriptbase':
		dataset_dir = sys.argv[2]
		run_scriptbase(dataset_dir)	

	elif len(sys.argv) > 1:
		run(sys.argv[1], dataset_dir)

	else:
		for key in movie_code:
			run(key, dataset_dir)
