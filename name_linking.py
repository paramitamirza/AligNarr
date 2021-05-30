import os
import re
import helper as pre
from fuzzywuzzy import process
from fuzzywuzzy import fuzz
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.cluster import DBSCAN
import spacy
from allennlp.predictors.predictor import Predictor
import allennlp_models.coref	

dir_path = os.path.dirname(os.path.realpath(__file__))
list_titles = pre.get_titles(dir_path+'/data/gazetteer/')

nlp = spacy.load("en_core_web_lg")
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz")

def get_entities(script_names, summary_names):

	### Entity linking: summary to script entities ###
	summary_del = set()
	for m in summary_names:
		fi_m = ' '.join([w for w in word_tokenize(m) if not w in pre.stop_words])

		names = list(script_names.keys())
		ratios = process.extractOne(fi_m, names)#, scorer=fuzz.token_set_ratio)
		ratios_high = [r for r in [ratios] if r[1] >= 80]
		if len(ratios_high) > 0:
			for (k, r) in ratios_high:
				#print("sum--", fi_m, "--", k, r)
				if k.count(' ') == 0 and fi_m.count(' ') > 0:
					if fuzz.token_set_ratio(k, fi_m) == 100 and not pre.longestSubstring(k, fi_m).strip().lower() in list_titles:
						script_names[m] = script_names[k]
				elif fi_m.count(' ') == 0 and k.count(' ') > 0:
					if fuzz.token_set_ratio(k, fi_m) == 100 and not pre.longestSubstring(k, fi_m).strip().lower() in list_titles:
						script_names[fi_m] = script_names[k]
				elif fi_m.count(' ') == 0 and k.count(' ') == 0:
					if r > 90 or (k.startswith(fi_m) and fi_m.lower() not in pre.forbidden_inclusion) or (fi_m.startswith(k) and k.lower() not in pre.forbidden_inclusion):
						script_names[fi_m] = script_names[k]
				else:
					if not pre.longestSubstring(k, fi_m).strip().lower() in list_titles:
						script_names[fi_m] = script_names[k]
				
			summary_del.add(fi_m)

	for d in summary_del: del summary_names[d]

	### Entity linking: script to script entities ###
	script_checked = set()
	script_changed = set()
	sorted_keys = sorted(script_names, key=len, reverse=True) 
	sim_matrix = np.zeros((len(sorted_keys), len(sorted_keys)))
	for i, n in enumerate(sorted_keys):
		fi_n = ' '.join([w for w in word_tokenize(n) if not w in pre.stop_words])
		script_checked.add(n)

		to_be_checked = list(set(sorted(script_names, key=len, reverse=True)) - script_checked)

		ratios = process.extract(fi_n, to_be_checked)#, scorer=fuzz.token_set_ratio)
		ratios_high = [r for r in ratios if r[1] >= 80]
		if len(ratios_high) > 0:
			for (k, r) in ratios_high:
				#print("sc--", fi_n, "--", k, r)
				kk = re.sub('-', ' - ', k)
				if kk.count(' ') == 0 and fi_n.count(' ') > 0:
					if fuzz.token_set_ratio(k, fi_n) == 100 and not pre.longestSubstring(kk, fi_n).strip().lower() in list_titles:						
						j = sorted_keys.index(k)
						sim_matrix[i, j] = 1.0
						sim_matrix[j, i] = 1.0
				elif fi_n.count(' ') == 0 and k.count(' ') > 0:
					if fuzz.token_set_ratio(k, fi_n) == 100 and not pre.longestSubstring(kk, fi_n).strip().lower() in list_titles:
						j = sorted_keys.index(k)
						sim_matrix[i, j] = 1.0
						sim_matrix[j, i] = 1.0
				elif fi_n.count(' ') == 0 and k.count(' ') == 0:
					if (kk.startswith(fi_n) and fi_n.lower() not in pre.forbidden_inclusion) or (fi_n.startswith(kk) and kk.lower() not in pre.forbidden_inclusion) or (kk.endswith(fi_n) and fi_n.lower() not in pre.forbidden_inclusion) or (fi_n.endswith(kk) and kk.lower() not in pre.forbidden_inclusion):
						j = sorted_keys.index(k)
						sim_matrix[i, j] = 1.0
						sim_matrix[j, i] = 1.0
					elif r > 90:
						j = sorted_keys.index(k)
						sim_matrix[i, j] = r/float(100)
						sim_matrix[j, i] = r/float(100)
				else:
					if not pre.longestSubstring(kk, fi_n).strip().lower() in list_titles:
						j = sorted_keys.index(k)
						sim_matrix[i, j] = r/float(100)
						sim_matrix[j, i] = r/float(100)


	### Check for first/last name of multiple people ###
	def is_ambiguous(name, entities):
		ambiguous_idx = set()
		if len(entities) > 1:
			for j, e in entities:
				for jj, ee in entities:
					if e != ee and e.count(' ') == ee.count(' '):
						if fuzz.ratio(e.replace(name, '').strip(), ee.replace(name, '').strip()) < 80:
							ambiguous_idx.add(j)
							ambiguous_idx.add(jj)
		return ambiguous_idx

	for i, m in enumerate(sorted_keys):
		if m.count(' ') == 0 and m != 'vitti':
			sim_names = [(j, n) for j, n in enumerate(sorted_keys) if sim_matrix[i, j] == 1.0]
			sim_names_clean = []
			for idx, name in sim_names:
				cleaned = ' '.join([w for w in word_tokenize(name) if not w in list_titles]).strip()
				if cleaned != m and cleaned is not "": sim_names_clean.append((idx, cleaned))
			for a in is_ambiguous(m, sim_names_clean):
				sim_matrix[i, a] = 0.0
				sim_matrix[a, i] = 0.0


	### Clustering ###
	dist_matrix = np.abs(np.subtract(sim_matrix, 1.0))
	db = DBSCAN(eps=0.1, min_samples=1, metric='precomputed').fit(dist_matrix)    #distance matrix
	#db = DBSCAN(eps=1, min_samples=3).fit(embedding_list)    
	core_samples = db.core_sample_indices_
	labels = db.labels_
	
	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
	clusters = []
	outliers = []
	for nc in range(n_clusters_):
		clusters.append([sorted_keys[k] for k, v in enumerate(labels) if v == nc])
	# Outliers!
	if -1 in labels:
		outliers = [sorted_keys[k] for k, v in enumerate(labels) if v == -1]

	for cluster in clusters:
		#print(cluster)
		rep_entity = script_names[cluster[0]]
		for key in cluster:
			script_names[key] = rep_entity
				
	entity_names = {}
	for k, v in script_names.items():
		entity_names[v] = entity_names.get(v, [])
		entity_names[v].append(k)

	return (script_names, summary_names, entity_names)

def resolve_allennlp(predictor, text, entity_rep_names):
	#print('### ', text)
	coref_result = predictor.predict(document=text)
	clusters = coref_result['clusters']
	doc = coref_result['document']

	singular_pronouns = ['he', 'He', 'she', 'She', 'it', 'It', 'I', 'you', 'You']
	plural_pronouns = ['they', 'They', 'we', 'We']
	entity_rep = list(entity_rep_names.values())

	for cl in clusters:
		spans = []
		spans_idx = []
		for span in cl:
			spans.append(' '.join(doc[span[0]:span[1]+1]))
			spans_idx.append(span[0])

		if any(item in spans for item in singular_pronouns):
			rep_name = ''
			for span in spans:
				if span in entity_rep:
					rep_name = span
					break
					
			if rep_name != '':
				for i, span in enumerate(spans):
					if span in entity_rep:
						rep_name = span
					elif span in singular_pronouns:
						doc[spans_idx[i]] = rep_name

		elif any(item in spans for item in plural_pronouns):
			rep_name = ''
			for span in spans:
				if ' and ' in span and all(item in entity_rep for item in [name.strip() for name in span.split(' and ')]): 
					rep_name = span
					break

			if rep_name != '':
				for i, span in enumerate(spans):
					if ' and ' in span and all(item in entity_rep for item in [name.strip() for name in span.split(' and ')]): 
						rep_name = span
					elif span in plural_pronouns:
						doc[spans_idx[i]] = rep_name

	#print('---> ', ' '.join(doc))
	return ' '.join(doc)

def replace_linked_entities(entity_names, names_original, script_sentences, stage_sentences, summary_sentences, coreference=True):
	entity_orig_names = {} 
	entity_rep_names = {}

	summary_for_coref = '\n'.join(summary_sentences)
	summary_replaced = '\n'.join(summary_sentences)
	summary_replaced_html = '\n'.join(summary_sentences)
	for k in entity_names:
		original_text = set()
		rep_name = k
		for name in entity_names[k]:
			for orig in names_original[name]: 
				original_text.add(orig)

		original_names = list(original_text)
		original_names.sort(key=len, reverse=True)

		for orig_text in original_names:
			if orig_text.isupper():
				if orig_text != 'RECIOUS':
					rep_name = orig_text

			if orig_text != "":
				try:
					summary_replaced = re.sub(r'\b%s\b' % orig_text, k, summary_replaced)
					summary_replaced = re.sub(r'\b%s\b' % orig_text.upper(), k, summary_replaced)
					summary_replaced = re.sub(r'\b%s\b' % orig_text.title(), k, summary_replaced)

					summary_replaced_html = re.sub(r'\b%s\b' % orig_text, '#'+k+'#', summary_replaced_html)
					summary_replaced_html = re.sub(r'\b%s\b' % orig_text.upper(), '#'+k+'#', summary_replaced_html)
					summary_replaced_html = re.sub(r'\b%s\b' % orig_text.title(), '#'+k+'#', summary_replaced_html)
				
				except re.error:
					print('***Cannot replace', orig_text)	

		if rep_name == k: 
			rep_name = original_names[-1]	
		rep_name = rep_name.replace('-', '')
		rep_name = rep_name.title()

		entity_rep_names[k] = rep_name		
		entity_orig_names[k] = original_names

	summary_for_coref = summary_replaced

	summary_entities = set()
	for k in entity_names:
		if re.search(r'\b%s\b' % k, summary_replaced): summary_entities.add(k)	

		summary_replaced_html = re.sub('#'+k+'#', '<a class='+k+' href="#" onclick="loadJSONEntity(this)">'+entity_rep_names[k]+'</a>', summary_replaced_html)
		summary_for_coref = re.sub(r'\b%s\b' % k, entity_rep_names[k], summary_for_coref)
				

	summary_replaced_sentences = summary_replaced.splitlines()
	#print(summary_replaced_sentences)

	if coreference:
		summary_replaced_sentences = []
		summary_replaced_html_sentences = []
		summary_to_be_resolved = '\n'.join([sent.split(' ### ')[1] for sent in summary_for_coref.splitlines()])
		#print('#####\n', summary_to_be_resolved)
		#summary_resolved = coref.resolve_spacy(nlp, summary_to_be_resolved)
		summary_resolved = resolve_allennlp(predictor, summary_to_be_resolved, entity_rep_names)
		#print('#####\n', summary_resolved)

		summary_resolved_html = summary_resolved
		#for k, rep_name in entity_rep_names.items():
		for k in sorted(entity_rep_names, key=lambda k: len(entity_rep_names[k]), reverse=True):
			try:
				summary_resolved = re.sub(r'\b%s\b' % entity_rep_names[k], k, summary_resolved)
				summary_resolved_html = re.sub(r'\b%s\b' % entity_rep_names[k], '<a class='+k+' href="#" onclick="loadJSONEntity(this)">'+entity_rep_names[k]+'</a>', summary_resolved_html)
			except re.error:
				print('***Cannot replace', entity_rep_names[k])

		for idx, sent_resolved in enumerate(summary_resolved.splitlines()):
			summary_replaced_sentences.append('s'+str(idx)+' ### '+sent_resolved)
			summary_replaced_html_sentences.append('s'+str(idx)+' ### '+summary_resolved_html.splitlines()[idx])

		summary_replaced_html = '\n'.join(summary_replaced_html_sentences)


	stage_replaced = '\n'.join(stage_sentences)
	stage_replaced_html = '\n'.join(stage_sentences)
	for k in entity_names:
		if k in summary_entities:
			original_text = set()
			rep_name = k
			for name in entity_names[k]:
				for orig in names_original[name]: 
					original_text.add(orig)

			original_names = list(original_text)
			original_names.sort(key=len, reverse=True)

			for orig_text in original_names:
				if orig_text != "":
					try:
						stage_replaced = re.sub(r'\b%s\b' % orig_text, k, stage_replaced)
						stage_replaced = re.sub(r'\b%s\b' % orig_text.upper(), k, stage_replaced)
						stage_replaced = re.sub(r'\b%s\b' % orig_text.title(), k, stage_replaced)

						stage_replaced_html = re.sub(r'\b%s\b' % orig_text, '#'+k+'#', stage_replaced_html)
						stage_replaced_html = re.sub(r'\b%s\b' % orig_text.upper(), '#'+k+'#', stage_replaced_html)
						stage_replaced_html = re.sub(r'\b%s\b' % orig_text.title(), '#'+k+'#', stage_replaced_html)

					except re.error:
						print('***Cannot replace', orig_text)	

	for k in entity_names:
		if k in summary_entities:
			stage_replaced_html =re.sub('#'+k+'#', '<a class='+k+' href="#" onclick="loadJSONEntity(this)">'+entity_rep_names[k]+'</a>', stage_replaced_html)		

	stage_replaced_sentences = stage_replaced.splitlines()
	#print(stage_replaced_sentences)

	script_for_coref = '\n'.join(script_sentences)
	script_replaced = '\n'.join(script_sentences)
	script_replaced_html = '\n'.join(script_sentences)
	for k in entity_names:
		#if len(entity_names[k]) > 1 or 
		if k in summary_entities:
			rep_name = k
			for name in entity_names[k]:
				for orig in names_original[name]: 
					original_text.add(orig)

			original_names = list(original_text)
			original_names.sort(key=len, reverse=True)

			for orig_text in original_names:
				try:
					script_replaced = re.sub(r'\b%s\b' % orig_text, k, script_replaced)
					script_replaced = re.sub(r'\b%s\b' % orig_text.upper(), k, script_replaced)
					script_replaced = re.sub(r'\b%s\b' % orig_text.title(), k, script_replaced)

					script_replaced_html = re.sub(r'\b%s\b' % orig_text, '#'+k+'#', script_replaced_html)
					script_replaced_html = re.sub(r'\b%s\b' % orig_text.upper(), '#'+k+'#', script_replaced_html)
					script_replaced_html = re.sub(r'\b%s\b' % orig_text.title(), '#'+k+'#', script_replaced_html)

				except re.error:
					print('***Cannot replace', orig_text)	

	script_for_coref = script_replaced

	for k in entity_names:
		if k in summary_entities:			
			script_replaced_html =re.sub('#'+k+'#', '<a class='+k+' href="#" onclick="loadJSONEntity(this)">'+entity_rep_names[k]+'</a>', script_replaced_html)
			script_for_coref = re.sub(r'\b%s\b' % k, entity_rep_names[k], script_for_coref)

	script_replaced_sentences = script_replaced.splitlines()
	#print(script_replaced_sentences)

	if False:
		import coref_resolution as coref

		script_replaced_sentences = []
		script_replaced_html_sentences = []

		description = []
		utterance = []
		description_ids = []
		utterance_ids = []

		p_desc = re.compile('s\d+d\d+')
		p_utt = re.compile('s\d+u\d+')
		prev_scene = ''
		for sent in script_for_coref.splitlines():
			sent_id = sent.split(' ### ')[0]
			scene_id = re.split('d|u', sent_id)[0]

			if scene_id != prev_scene:
				if description != []:
					description_to_be_resolved = '\n'.join(description)
					description_resolved = description_to_be_resolved
					if len(description) > 1:
						description_resolved = resolve_allennlp(predictor, description_to_be_resolved, entity_rep_names)

					for d, desc in enumerate(description_resolved.splitlines()):
						script_replaced_sentences.append(description_ids[d] + ' ### ' + desc)

					description = []
					description_ids = []

				if utterance != []:
					utterance_to_be_resolved = '\n'.join(utterance)
					utterance_resolved = utterance_to_be_resolved
					if len(utterance) > 1:
						utterance_resolved = resolve_allennlp(predictor, utterance_to_be_resolved, entity_rep_names)

					for u, utt in enumerate(utterance_resolved.splitlines()):
						script_replaced_sentences.append(utterance_ids[u] + ' ### ' + utt)

					utterance = []
					utterance_ids = []

				if p_utt.match(sent_id):
					utterance.append(sent.split(' ### ')[1] + ' said " ' + sent.split(' ### ')[2].strip() + ' "')
					utterance_ids.append(sent_id)

				elif p_desc.match(sent_id):
					description.append(sent.split(' ### ')[1].strip())
					description_ids.append(sent_id)
			else:

				if p_utt.match(sent_id):
					if description != []:
						description_to_be_resolved = '\n'.join(description)
						description_resolved = description_to_be_resolved
						if len(description) > 1:
							description_resolved = resolve_allennlp(predictor, description_to_be_resolved, entity_rep_names)

						for d, desc in enumerate(description_resolved.splitlines()):
							script_replaced_sentences.append(description_ids[d] + ' ### ' + desc)

						description = []
						description_ids = []

					utterance.append(sent.split(' ### ')[1] + ' said " ' + sent.split(' ### ')[2].strip() + ' "')
					utterance_ids.append(sent_id)

				elif p_desc.match(sent_id):
					if utterance != []:
						utterance_to_be_resolved = '\n'.join(utterance)
						utterance_resolved = utterance_to_be_resolved
						if len(utterance) > 1:
							utterance_resolved = resolve_allennlp(predictor, utterance_to_be_resolved, entity_rep_names)

						for u, utt in enumerate(utterance_resolved.splitlines()):
							script_replaced_sentences.append(utterance_ids[u] + ' ### ' + utt)

						utterance = []
						utterance_ids = []

					description.append(sent.split(' ### ')[1].strip())
					description_ids.append(sent_id)

			prev_scene = scene_id

		if description != []:
			description_to_be_resolved = '\n'.join(description)
			description_resolved = description_to_be_resolved
			if len(description) > 1:
				description_resolved = resolve_allennlp(predictor, description_to_be_resolved, entity_rep_names)

			for d, desc in enumerate(description_resolved.splitlines()):
				script_replaced_sentences.append(description_ids[d] + ' ### ' + desc)

		if utterance != []:
			utterance_to_be_resolved = '\n'.join(utterance)
			utterance_resolved = utterance_to_be_resolved
			if len(utterance) > 1:
				utterance_resolved = resolve_allennlp(predictor, utterance_to_be_resolved, entity_rep_names)

			for u, utt in enumerate(utterance_resolved.splitlines()):
				script_replaced_sentences.append(utterance_ids[u] + ' ### ' + utt)

	return (entity_orig_names, entity_rep_names, summary_replaced_sentences, stage_replaced_sentences, script_replaced_sentences, summary_replaced_html, stage_replaced_html, script_replaced_html)