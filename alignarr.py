from __future__ import print_function
import sys
import os
#import preprocess as pre
#import name_linking as link
#import helper as he
import evaluation as ev
import re
import numpy as np
import string
import json
import pickle
import math
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
from gurobipy import *
import torch
from sklearn.metrics.pairwise import cosine_similarity
import time
import csv

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

dir_path = os.path.dirname(os.path.realpath(__file__))
dataset_dir = dir_path + '/dataset/10_movies/'
gold_alignment_dir = dir_path + '/data/gold_alignment/'
w2v_bin_file = '/home/paramita/Tools/GoogleNews-vectors-negative300.bin'

movie_list = ['anastasia', 'cars', 'shrek', 'south', 'swordfish', 'butterfly', 'silence', 'walle', 'pulp', 'cast']

movie_code = {'anastasia': 'Anastasia', 
				'cars': 'Cars_2',  
				'shrek': 'Shrek', 
				'south': 'South_Park', 
				'swordfish': 'Swordfish', 
				'butterfly': 'The_Butterfly_Effect', 
				'silence': 'The_Silence_of_the_Lambs', 
				'walle': 'WALL-E',
				'pulp': 'Pulp_Fiction',
				'cast': 'Cast_Away'
				}

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))	

def preprocess_for_alignment(summary_replaced_sentences, stage_replaced_sentences, script_replaced_sentences, num_scenes, num_summary_sentences, entity_removal, lemmatized):
	script_scenes_nostopword = [''] * num_scenes					#stopword removed
	summary_sentences_nostopword = [''] * num_summary_sentences 	#stopword removed

	script_scenes = [''] * num_scenes
	summary_sentences = [''] * num_summary_sentences

	script_entities_str = [''] * num_scenes
	summary_entities_str = [''] * num_summary_sentences

	script_entities = []
	summary_entities = []
	
	for i, sent in enumerate(stage_replaced_sentences):
		sent_text = stage_replaced_sentences[i].split(' ### ')[1] + ' . '

		if entity_removal:
			sent_text = re.sub(r'E\d+', '', sent_text)
		
		for e in set(re.findall(r'E\d+', sent_text)):
			script_entities_str[i] += ' ' + e

		sent_text = sent_text.translate(str.maketrans('', '', string.punctuation))
		if lemmatized:
			sent_text_nostopword = ' '.join([lemmatizer.lemmatize(w).lower() for w in word_tokenize(sent_text) if not w.lower() in stop_words])
			sent_text = ' '.join([lemmatizer.lemmatize(w).lower() for w in word_tokenize(sent_text)])
		else:
			sent_text_nostopword = ' '.join([w.lower() for w in word_tokenize(sent_text) if not w.lower() in stop_words])
			sent_text = ' '.join([w.lower() for w in word_tokenize(sent_text)])

		script_scenes_nostopword[i] += sent_text_nostopword + '\n'
		script_scenes[i] += sent_text + '\n'

	for sent in script_replaced_sentences:
		scene_id = re.split('d|u', sent.split(' ### ')[0])[0]
		scene_idx = int(scene_id.replace('s', ''))		

		if entity_removal:
			sent_text = ' '.join(sent.split(' ### ')[1:])
			sent_text = re.sub(r'E\d+', '', sent_text)
		else:
			sent_text = ' said '.join(sent.split(' ### ')[1:])

		for e in set(re.findall(r'E\d+', sent_text)):
			script_entities_str[scene_idx] += ' ' + e

		sent_text = sent_text.translate(str.maketrans('', '', string.punctuation))
		if lemmatized:
			sent_text_nostopword = ' '.join([lemmatizer.lemmatize(w).lower() for w in word_tokenize(sent_text) if not w.lower() in stop_words])
			sent_text = ' '.join([lemmatizer.lemmatize(w).lower() for w in word_tokenize(sent_text)])
		else:
			sent_text_nostopword = ' '.join([w.lower() for w in word_tokenize(sent_text) if not w.lower() in stop_words])
			sent_text = ' '.join([w.lower() for w in word_tokenize(sent_text)])

		script_scenes_nostopword[scene_idx] += sent_text_nostopword + '\n'
		script_scenes[scene_idx] += sent_text + '\n'

	for j, sent in enumerate(summary_replaced_sentences):
		sent_id = sent.split(' ### ')[0]
		sent_idx = int(sent_id.replace('s', ''))
		sent_text = sent.split(' ### ')[1]

		if entity_removal:
			sent_text = re.sub(r'E\d+', '', sent_text)

		for e in set(re.findall(r'E\d+', sent_text)):
			summary_entities_str[sent_idx] += ' ' + e

		sent_text = sent_text.translate(str.maketrans('', '', string.punctuation))
		if lemmatized:
			sent_text_nostopword = ' '.join([lemmatizer.lemmatize(w).lower() for w in word_tokenize(sent_text) if not w.lower() in stop_words])
			sent_text = ' '.join([lemmatizer.lemmatize(w).lower() for w in word_tokenize(sent_text)])
		else:
			sent_text_nostopword = ' '.join([w.lower() for w in word_tokenize(sent_text) if not w.lower() in stop_words])
			sent_text = ' '.join([w.lower() for w in word_tokenize(sent_text)])

		summary_sentences_nostopword[sent_idx] += sent_text_nostopword
		summary_sentences[sent_idx] += sent_text

	for i, sent in enumerate(stage_replaced_sentences):
		script_entities.append(set(script_entities_str[i].strip().split(' ')))

	for j, sent in enumerate(summary_replaced_sentences):
		summary_entities.append(set(summary_entities_str[j].strip().split(' ')))

	return (script_scenes, summary_sentences, script_scenes_nostopword, summary_sentences_nostopword, script_entities, summary_entities)


def preprocess(code, dataset_dir, coreference, entity_removal, lemmatized):
	import preprocess as pre
	import name_linking as link

	script_xml = dataset_dir+'/' + movie_code[code] + '/script.xml'
	(wikiplot_sentences, _) = pre.get_wiki_plot(movie_code[code], dataset_dir+'/' + movie_code[code] + '/wikiplot.txt')

	(script_names, summary_names, names_original, script_sentences, stage_sentences, summary_sentences, num_scenes, num_summary_sentences) = pre.process_script_and_summary(script_xml, wikiplot_sentences)
	(script_names, summary_names, entity_names) = link.get_entities(script_names, summary_names)
	(entity_orig_names, entity_rep_names, summary_replaced_sentences, stage_replaced_sentences, script_replaced_sentences, summary_replaced_html, stage_replaced_html, script_replaced_html) = link.replace_linked_entities(entity_names, names_original, script_sentences, stage_sentences, summary_sentences, coreference=coreference)

	(script_scenes, summary_sentences, script_scenes_nostopword, summary_sentences_nostopword, script_entities, summary_entities) = preprocess_for_alignment(summary_replaced_sentences, stage_replaced_sentences, script_replaced_sentences, num_scenes, num_summary_sentences, entity_removal, lemmatized)

	return (script_scenes, summary_sentences, script_scenes_nostopword, summary_sentences_nostopword, script_entities, summary_entities)

def preprocess_scriptbase(movie_title, dataset_dir, coreference, entity_removal, lemmatized):
	import preprocess as pre
	import name_linking as link

	script_xml = dataset_dir+'/' + movie_title + '/script.xml'
	(wikiplot_sentences, _) = pre.get_wiki_plot(movie_title, dataset_dir+'/' + movie_title + '/wikiplot.txt')

	(script_names, summary_names, names_original, script_sentences, stage_sentences, summary_sentences, num_scenes, num_summary_sentences) = pre.process_script_and_summary(script_xml, wikiplot_sentences)
	(script_names, summary_names, entity_names) = link.get_entities(script_names, summary_names)
	(entity_orig_names, entity_rep_names, summary_replaced_sentences, stage_replaced_sentences, script_replaced_sentences, summary_replaced_html, stage_replaced_html, script_replaced_html) = link.replace_linked_entities(entity_names, names_original, script_sentences, stage_sentences, summary_sentences, coreference=coreference)

	with open('alignarr_scriptbase_statistic.csv', 'a', newline='') as csvfile:
		ratio = num_scenes / float(num_summary_sentences)
		writer = csv.writer(csvfile)
		writer.writerow([movie_title, num_scenes, num_summary_sentences, ratio])

	### Write files for annotation ###
	pre.write_alignarr_files(dataset_dir, movie_title, summary_replaced_html, stage_replaced_html, script_replaced_html, entity_orig_names, num_scenes, num_summary_sentences)
	#pre.write_resolved_summary(dataset_dir, movie_title, summary_replaced_sentences, entity_rep_names)

	(script_scenes, summary_sentences, script_scenes_nostopword, summary_sentences_nostopword, script_entities, summary_entities) = preprocess_for_alignment(summary_replaced_sentences, stage_replaced_sentences, script_replaced_sentences, num_scenes, num_summary_sentences, entity_removal, lemmatized)

	return (script_scenes, summary_sentences, script_scenes_nostopword, summary_sentences_nostopword, script_entities, summary_entities)

def get_similarity_bert_score(sim_matrix_dir, code, script_scenes, summary_sentences, load=True):
	#https://github.com/Tiiiger/bert_score
	if not os.path.exists(sim_matrix_dir): os.makedirs(sim_matrix_dir)

	if os.path.isfile(sim_matrix_dir + 'bert_score_' + code + '.sim') and load:
		with open(sim_matrix_dir + 'bert_score_' + code + '.sim', 'rb') as fp:
			similarity_matrix = pickle.load(fp)
			return similarity_matrix
	else:
		with open(sim_matrix_dir + 'bert_score_' + code + '.sim', 'wb') as fp:
			
			from bert_score import score

			similarity_matrix = np.zeros((len(script_scenes), len(summary_sentences)))

			for i, scene in enumerate(script_scenes):
				for j, sent in enumerate(summary_sentences):
					scene_sentences = scene.splitlines()
					summary_repeat = []
					for n in scene_sentences: summary_repeat.append(sent)
					P, R, F1 = score(scene_sentences, summary_repeat, idf=True, lang="en")
					similarity_matrix[i, j] = P.max()

					#print(i, j, similarity_matrix[i, j])

			pickle.dump(similarity_matrix,fp)
			return similarity_matrix

def get_bert_token_embeddings(model, tokenizer, sent):
	marked_sent = "[CLS] " + sent + " [SEP]"
	tokenized_sent = tokenizer.tokenize(marked_sent)

	indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_sent)
	segments_ids = [1] * len(tokenized_sent)
	tokens_tensor = torch.tensor([indexed_tokens])
	segments_tensors = torch.tensor([segments_ids])

	# If you have a GPU, put everything on cuda
	tokens_tensor = tokens_tensor.to('cuda')
	segments_tensors = segments_tensors.to('cuda')
	model.to('cuda')

	with torch.no_grad():
		outputs = model(tokens_tensor, segments_tensors)

		# Evaluating the model will return a different number of objects based on 
		# how it's  configured in the `from_pretrained` call earlier. In this case, 
		# becase we set `output_hidden_states = True`, the third item will be the 
		# hidden states from all layers. See the documentation for more details:
		# https://huggingface.co/transformers/model_doc/bert.html#bertmodel
		hidden_states = outputs[2]

	token_embeddings = torch.stack(hidden_states, dim=0)
	token_embeddings = torch.squeeze(token_embeddings, dim=1)
	token_embeddings = token_embeddings.permute(1,0,2)

	token_vecs_sum = []
	for token in token_embeddings:
		# Sum the vectors from the last four layers.
		sum_vec = torch.sum(token[-4:], dim=0)

		# Use `sum_vec` to represent `token`.
		token_vecs_sum.append(sum_vec)

	return (tokenized_sent, token_vecs_sum)

def is_content_token(i, tokenized_text):
	p = re.compile('##\d+')
	tok = tokenized_text[i]
	if p.match(tok) or (tok == 'e' and p.match(tokenized_text[i+1])):
		return False
	elif tok in stop_words:
		return False
	elif tok == 'said' or tok == '[SEP]' or tok == '[CLS]':
		return False

	return True

def get_similarity_bert(sim_matrix_dir, code, script_scenes, summary_sentences, load=True):
	#https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/#3-extracting-embeddings
	if not os.path.exists(sim_matrix_dir): os.makedirs(sim_matrix_dir)

	if os.path.isfile(sim_matrix_dir + 'bert_' + code + '.sim') and load:
		with open(sim_matrix_dir + 'bert_' + code + '.sim', 'rb') as fp:
			similarity_matrix = pickle.load(fp)
			return similarity_matrix
	else:
		with open(sim_matrix_dir + 'bert_' + code + '.sim', 'wb') as fp:
			
			from transformers import BertTokenizer, BertModel

			similarity_matrix = np.zeros((len(script_scenes), len(summary_sentences)))

			# Load pre-trained model tokenizer (vocabulary)
			tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

			# Load pre-trained model (weights)
			model = BertModel.from_pretrained('bert-large-uncased',
											  output_hidden_states = True, # Whether the model returns all hidden-states.
											  )

			# Put the model in "evaluation" mode, meaning feed-forward operation.
			model.eval()

			for i, scene in enumerate(script_scenes):
				for j, sent in enumerate(summary_sentences):
					scene_embeddings = []
					for scene_sent in scene.splitlines():
						(tokenized_scene_sent, token_emb_scene_sent) = get_bert_token_embeddings(model, tokenizer, scene_sent)

						for e, emb in enumerate(token_emb_scene_sent):
							if is_content_token(e, tokenized_scene_sent):
								scene_embeddings.append(emb)

					(tokenized_sent, token_emb_sent) = get_bert_token_embeddings(model, tokenizer, sent)
					for e, emb in enumerate(token_emb_sent):
						if is_content_token(e, tokenized_sent):

							#Tokens to compare between summary and scene are only content words
							#print(tokenized_sent[e], [tok for (tok, emb) in scene_embeddings])

							embeddings = [emb]
							embeddings += scene_embeddings
							cosine_sim = cosine_similarity(torch.stack(embeddings).cpu().detach().numpy())[0][1:]

							try:
								#if np.max(cosine_sim) > 0.5:
								if np.max(cosine_sim) > 0.7: 
									similarity_matrix[i, j] += np.max(cosine_sim)
							except ValueError:  #raised if 'cosine_sim' is empty.
								pass

					#print(i, j, similarity_matrix[i, j])

			pickle.dump(similarity_matrix,fp)
			return similarity_matrix

def get_similarity_sts(sim_matrix_dir, code, script_scenes, summary_sentences, load=True):
	#sts: https://github.com/pdrm83/sts 
	#sentence-transformers: https://github.com/UKPLab/sentence-transformers 
	if not os.path.exists(sim_matrix_dir): os.makedirs(sim_matrix_dir)

	if os.path.isfile(sim_matrix_dir + 'sts_' + code + '.sim') and load:
		with open(sim_matrix_dir + 'sts_' + code + '.sim', 'rb') as sts:
			similarity_matrix_sts = pickle.load(sts)
			return similarity_matrix_sts
	else:
		with open(sim_matrix_dir + 'w2v_' + code + '.sim', 'rb') as w2v, open(sim_matrix_dir + 'entity_' + code + '.sim', 'rb') as ent, open(sim_matrix_dir + 'sts_' + code + '.sim', 'wb') as sts:
			
			from sentence_transformers import SentenceTransformer

			similarity_matrix_w2v = np.add(pickle.load(w2v), pickle.load(ent))
			similarity_matrix_sts = np.zeros((len(script_scenes), len(summary_sentences)))

			model = SentenceTransformer('stsb-roberta-large')
			
			for i, scene in enumerate(script_scenes):
				for j, sent in enumerate(summary_sentences):
					if similarity_matrix_w2v[i, j] > 0:
						sentences = [sent]
						sentences += scene.splitlines()

						#sent2vec
						#vectorizer = Vectorizer()
						#vectorizer.bert(sentences)
						#vectors_bert = vectorizer.vectors

						#sentence-transformers
						vectors_bert = model.encode(sentences)

						cosine_sim = cosine_similarity(vectors_bert)[0][1:]
						similarity_matrix_sts[i, j] = np.max(cosine_sim)

					#print(i, j, similarity_matrix_sts[i, j])

			pickle.dump(similarity_matrix_sts,sts)
			return similarity_matrix_sts

def get_similarity_sts_nli(sim_matrix_dir, code, script_scenes, summary_sentences, load=True):
	#sts: https://github.com/pdrm83/sts 
	#sentence-transformers: https://github.com/UKPLab/sentence-transformers 
	if not os.path.exists(sim_matrix_dir): os.makedirs(sim_matrix_dir)

	if os.path.isfile(sim_matrix_dir + 'nli_' + code + '.sim') and os.path.isfile(sim_matrix_dir + 'sts_' + code + '.sim') and load:
		with open(sim_matrix_dir + 'nli_' + code + '.sim', 'rb') as nli, open(sim_matrix_dir + 'sts_' + code + '.sim', 'rb') as sts:
			similarity_matrix_nli = pickle.load(nli)
			similarity_matrix_sts = pickle.load(sts)
			return (similarity_matrix_sts, similarity_matrix_nli)
	else:
		with open(sim_matrix_dir + 'w2v_' + code + '.sim', 'rb') as w2v, open(sim_matrix_dir + 'nli_' + code + '.sim', 'wb') as nli, open(sim_matrix_dir + 'sts_' + code + '.sim', 'wb') as sts:
			
			from sentence_transformers import SentenceTransformer
			from allennlp.predictors.predictor import Predictor
			import allennlp_models.pair_classification

			similarity_matrix_w2v = pickle.load(w2v)
			similarity_matrix_nli = np.zeros((len(script_scenes), len(summary_sentences)))
			similarity_matrix_sts = np.zeros((len(script_scenes), len(summary_sentences)))

			model = SentenceTransformer('stsb-roberta-large')
			predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/snli_roberta-2020.06.09.tar.gz", "textual_entailment")

			for i, scene in enumerate(script_scenes):
				for j, sent in enumerate(summary_sentences):
					if similarity_matrix_w2v[i, j] > 0:
						sentences = [sent]
						sentences += scene.splitlines()

						#sent2vec
						#vectorizer = Vectorizer()
						#vectorizer.bert(sentences)
						#vectors_bert = vectorizer.vectors

						#sentence-transformers
						vectors_bert = model.encode(sentences)

						cosine_sim = cosine_similarity(vectors_bert)[0][1:]
						similarity_matrix_sts[i, j] = np.max(cosine_sim)

						#get top 5 similar to compute entailment
						entailment = 0
						for k in cosine_sim.argsort()[-5:][::-1]:
							#print(cosine_sim[k], sent, scene.splitlines()[k])
							entailment += predictor.predict(sent, scene.splitlines()[k])['probs'][0]

						#similarity_matrix[i, j] = entailment / len(scene_sentences)
						similarity_matrix_nli[i, j] = entailment

					#print(i, j, similarity_matrix_sts[i, j], similarity_matrix_nli[i, j])

			pickle.dump(similarity_matrix_sts,sts)
			pickle.dump(similarity_matrix_nli,nli)
			return (similarity_matrix_sts, similarity_matrix_nli)

def get_similarity_w2v(sim_matrix_dir, code, script_scenes, summary_sentences, script_entities, summary_entities, load=True):
	if not os.path.exists(sim_matrix_dir): os.makedirs(sim_matrix_dir)

	if os.path.isfile(sim_matrix_dir + 'w2v_' + code + '.sim') and load:
		with open(sim_matrix_dir + 'w2v_' + code + '.sim', 'rb') as fp:
			similarity_matrix = pickle.load(fp)
			return similarity_matrix
	else:
		with open(sim_matrix_dir + 'w2v_' + code + '.sim', 'wb') as out:

			from sklearn.feature_extraction.text import CountVectorizer
			import gensim
			from gensim.models import KeyedVectors
			import networkx as nx

			sentence_pairs = []
			index_pairs = []	
			
			vocabs = {}
			sentences = summary_sentences + script_scenes

			vect = CountVectorizer(min_df=1, ngram_range=(1, 1), max_features=5000)
			vect = CountVectorizer()
			corpus_vect = vect.fit_transform(sentences)
			#corpus_vect_gensim = gensim.matutils.Sparse2Corpus(corpus_vect, documents_columns=False)
			#count_vect = [sum(x) for x in zip(*corpus_vect.toarray())]

			if not os.path.exists(dir_path + '/data/w2v_vocab/'): os.makedirs(dir_path + '/data/w2v_vocab/')
			with open(dir_path + '/data/w2v_vocab/'+code+'.v', 'w') as f:
				for key, val in vect.vocabulary_.items():
					f.write(key + '\n')
			
			g = nx.Graph()
			model = KeyedVectors.load_word2vec_format(w2v_bin_file, binary=True)

			sim_pairs = {}
			if os.path.isfile(dir_path + '/data/w2v_pairs/' + code + '.sim') and load:
				with open(dir_path + '/data/w2v_pairs/'+code+'.sim', 'rb') as fp:
					sim_pairs = pickle.load(fp)
			else:
				with open(dir_path + '/data/w2v_vocab/'+code+'.v', 'r') as fin:
					words = [w.strip() for w in fin.readlines()]
					
					for i in range(len(words)):
						for j in range(i+1, len(words)):
							if words[i] in model.vocab and words[j] in model.vocab:
								sim = model.similarity(words[i], words[j])
								if sim > 0.5:
									sim_pairs[words[i]+' '+words[j]] = sim

				if not os.path.exists(dir_path + '/data/w2v_pairs/'): os.makedirs(dir_path + '/data/w2v_pairs/')
				with open(dir_path + '/data/w2v_pairs/'+code+'.sim', 'wb') as fp:
					pickle.dump(sim_pairs, fp, protocol=pickle.HIGHEST_PROTOCOL)

			for pair in sim_pairs:
				g.add_edge(pair.split(' ')[0], pair.split(' ')[1], weight = sim_pairs[pair])

			for j, sent in enumerate(summary_sentences):
				for word in re.split('\s', sent):
					if word in vect.vocabulary_: g.add_edge('SU###'+str(j), word, weight = 1)

			for i, scene in enumerate(script_scenes):
				for word in re.split('\s', scene):
					if word in vect.vocabulary_: g.add_edge('SC###'+str(i), word, weight = 1)

			node_centrality = nx.betweenness_centrality(g)
			path_matrix = np.zeros((len(script_scenes), len(summary_sentences)))

			for i, x in enumerate(nx.connected_components(g)):
				connected = ' '.join(list(x))
				if 'SU###' in connected and 'SC###' in connected:
					summary_nodes = re.findall('SU###\d+', connected)
					scene_nodes = re.findall('SC###\d+', connected)

					for su in summary_nodes:
						su_idx = int(su.split('###')[1])

						for sc in scene_nodes:
							sc_idx = int(sc.split('###')[1])

							#overlapping = summary_entities[su_idx].intersection(script_entities[sc_idx])

							paths = nx.all_simple_paths(g, source=su, target=sc, cutoff=3)
							for path in paths:
								if len(path) == 3:
									path_matrix[sc_idx, su_idx] += 1
								elif len(path) == 4:
									path_matrix[sc_idx, su_idx] += g[path[1]][path[2]]['weight']

			pickle.dump(path_matrix,out)
			return path_matrix

def get_similarity_bm25(sim_matrix_dir, code, script_scenes, summary_sentences, script_entities, summary_entities, load=True):
	if not os.path.exists(sim_matrix_dir): os.makedirs(sim_matrix_dir)

	if os.path.isfile(sim_matrix_dir + 'bm25_' + code + '.sim') and load:
		with open(sim_matrix_dir + 'bm25_' + code + '.sim', 'rb') as fp:
			similarity_matrix = pickle.load(fp)
			return similarity_matrix
	else:
		with open(sim_matrix_dir + 'bm25_' + code + '.sim', 'wb') as fp:
			
			from rank_bm25 import BM25Okapi

			similarity_matrix = np.zeros((len(script_scenes), len(summary_sentences)))

			tokenized_corpus = [re.split('\s', doc.strip()) for doc in script_scenes]
			bm25 = BM25Okapi(tokenized_corpus)

			prev_i = 0
			for j, sent in enumerate(summary_sentences):
				tokenized_query = re.split('\s', sent)
				doc_scores = bm25.get_scores(tokenized_query)
				#sum_scores = np.sum(np.array(doc_scores))
				for i, score in enumerate(doc_scores):
					similarity_matrix[i, j] = score

			pickle.dump(similarity_matrix,fp)
			return similarity_matrix

def jaccard_similarity(s1, s2):
    return float(len(s1.intersection(s2)) / len(s1.union(s2)))

def get_similarity_entity(sim_matrix_dir, code, script_entities, summary_entities, num_scenes, num_summary_sentences, load=True):
	if not os.path.exists(sim_matrix_dir): os.makedirs(sim_matrix_dir)

	if os.path.isfile(sim_matrix_dir + 'entity_' + code + '.sim') and load:
		with open(sim_matrix_dir + 'entity_' + code + '.sim', 'rb') as fp:
			similarity_matrix = pickle.load(fp)
			return similarity_matrix
	else:
		with open(sim_matrix_dir + 'entity_' + code + '.sim', 'wb') as fp:
			similarity_matrix = np.zeros((num_scenes, num_summary_sentences))

			for i in range(num_scenes):
				for j in range(num_summary_sentences):
					overlapping = summary_entities[j].intersection(script_entities[i])
					if len(overlapping) > 0:
						similarity_matrix[i, j] += len(overlapping)
						#similarity_matrix[i, j] += (len(overlapping) * ((len(overlapping) / len(script_entities)) * (len(overlapping) / len(summary_entities))))
						#similarity_matrix[i, j] = jaccard_similarity(summary_entities[j], script_entities[i])

			pickle.dump(similarity_matrix,fp)
			return similarity_matrix

def get_alignment(output_dir, log_file, code, similarity_matrix, top_n, ensure_upward, limit_candidate=True, percentile=50):
	script_scenes = {}
	summary_sentences = {}

	(num_scenes, num_summary_sentences) = similarity_matrix.shape
	ratio = num_scenes / float(num_summary_sentences)
	ratio = math.ceil(ratio)

	if limit_candidate:
		for i in range(num_scenes):
			for j in range(num_summary_sentences):
				i_100 = (i/float(num_scenes)) * 100
				j_100 = (j/float(num_summary_sentences)) * 100
				if j_100 < i_100 + 30 and i_100 < j_100 + 30:
					similarity_matrix[i, j] = similarity_matrix[i, j]
				else:
					similarity_matrix[i, j] = 0

	prev_min_i = 0
	prev_max_i = 0
	prev_i = 0

	for j in range(num_summary_sentences):
		summary_sentences['sent'+str(j)] = set()
		
		if ensure_upward:
			for n in range(prev_min_i - ratio):
				similarity_matrix[n, j] = 0.0
			#if prev_max_i > 0:
			#	for n in range(prev_max_i + (3 * ratio), num_scenes):
			#		similarity_matrix[n, j] = 0.0

		sim_non_zero = [x for x in similarity_matrix[:, j] if x > 0]
		#sim_above_std = [x for x in sim_non_zero if x > np.std(sim_non_zero)]

		if len(sim_non_zero) > 0:
			
			ranked = similarity_matrix[:, j].argsort()[-top_n:][::-1]
			top_sim = similarity_matrix[ranked[0], j]
			if percentile > 0:
				threshold = np.percentile(sim_non_zero, percentile)
				filtered_ranked = [x for x in ranked if similarity_matrix[x, j] > 0 and similarity_matrix[x, j] >= threshold and (top_sim - similarity_matrix[x, j]) < np.std(sim_non_zero)]
			else:
				filtered_ranked = [x for x in ranked if similarity_matrix[x, j] > 0 and (top_sim - similarity_matrix[x, j]) < np.std(sim_non_zero)]

			if len(filtered_ranked) > 0:
				sorted_ranked = sorted(filtered_ranked)
				min_i_ranked = sorted_ranked[0]
				if ensure_upward:
					sorted_ranked = [x for x in filtered_ranked if x < (min_i_ranked + 5)]

				#aligned_i = []
				for i in sorted_ranked:
					#if similarity_matrix[i, j] > 0 and top_sim - similarity_matrix[i, j] < np.std(sim_non_zero):
					#print (j, i, similarity_matrix[i, j], threshold, np.std(sim_non_zero))

					summary_sentences['sent'+str(j)].add('scene'+str(i))

					if 'scene'+str(i) not in script_scenes: script_scenes['scene'+str(i)] = set()
					script_scenes['scene'+str(i)].add('sent'+str(j))
					
					#aligned_i.append(i)
					prev_i = i

				#if len(aligned_i) > 0: 
				prev_min_i = min(sorted_ranked)
				prev_max_i = max(sorted_ranked)

	if not os.path.exists(os.path.join(output_dir, log_file, '')): os.makedirs(os.path.join(output_dir, log_file, ''))
	with open(os.path.join(output_dir, log_file, code+'.json'), 'w') as outfile:
		obj = {}
		obj['scene'] = {}
		obj['summary'] = {}
		for scene in script_scenes:
			obj['scene'][scene] = list(script_scenes[scene])
		for sent in summary_sentences:
			obj['summary'][sent] = list(summary_sentences[sent])  
		obj['num_scenes'] = num_scenes
		obj['num_summary_sentences'] = num_summary_sentences
		json.dump(obj, outfile)

def get_ilp_alignment(output_dir, log_file, code, similarity_matrix, limit_candidate=True, percentile=50):
	script_scenes = {}
	summary_sentences = {}

	(num_scenes, num_summary_sentences) = similarity_matrix.shape
	ratio = num_scenes / float(num_summary_sentences)
	ratio = math.ceil(ratio)

	if limit_candidate:
		for i in range(num_scenes):
			for j in range(num_summary_sentences):
				i_100 = (i/float(num_scenes)) * 100
				j_100 = (j/float(num_summary_sentences)) * 100
				if j_100 < i_100 + 30 and i_100 < j_100 + 30:
					similarity_matrix[i, j] = similarity_matrix[i, j]
				else:
					similarity_matrix[i, j] = 0

	prev_min_i = 0
	prev_max_i = 0
	prev_i = 0
	filtered_sim = np.zeros((num_scenes, num_summary_sentences))
	for j in range(num_summary_sentences):
		sim_non_zero = [x for x in similarity_matrix[:, j] if x > 0]
		if len(sim_non_zero) > 0:
			ranked = similarity_matrix[:, j].argsort()[:][::-1]
			top_sim = similarity_matrix[ranked[0], j]
			if percentile > 0:
				threshold = np.percentile(sim_non_zero, percentile)
				filtered_ranked = [x for x in ranked if similarity_matrix[x, j] > 0 and similarity_matrix[x, j] >= threshold and (top_sim - similarity_matrix[x, j]) < np.std(sim_non_zero)]
			else:
				filtered_ranked = [x for x in ranked if similarity_matrix[x, j] > 0 and (top_sim - similarity_matrix[x, j]) < np.std(sim_non_zero)]
			for i in filtered_ranked:
				filtered_sim[i, j] = similarity_matrix[i, j]


	model = Model()
	Sc = range(num_scenes)
	Sm = range(num_summary_sentences)
	pairs = [(i,j) for i in Sc for j in Sm]
	x = model.addVars(pairs, vtype=GRB.BINARY, name='x')    # decision variable

	model.setObjective(quicksum(x[i,j]*filtered_sim[i, j] for i in Sc for j in Sm), GRB.MAXIMIZE)     # objective function

	# constraints
	max = 5
	#if ratio > 5: max = ratio

	model.addConstrs(x.sum('*',j) >= 1 for j in Sm)				# each summary must be paired with at least one scene
	model.addConstrs(x.sum('*',j) <= max for j in Sm)			# each summary can only be aligned with at most 5/ratio scenes
	
	model.addConstrs((x[i,j] + x[k,j]) <= 1 for i in Sc for j in Sm for k in Sc if k >= (i + max))								# pairwise constraint
	model.addConstrs((x[i,j] + x[k,j]) <= 1 for i in Sc for j in Sm for k in Sc if k <= (i - max) if i >= max)					# pairwise constraint
	model.addConstrs((x[i,j] + x[k,j-1]) <= 1 for i in Sc for j in Sm for k in Sc if k > i if j > 0)							# pairwise constraint
	model.addConstrs((x[i,j] + x[k,j+1]) <= 1 for i in Sc for j in Sm for k in Sc if k < i if j < (num_summary_sentences-1))	# pairwise constraint

	#model.setParam("MIPGap", 0.05)
	model.setParam('TimeLimit', 7200)
	model.optimize()

	alignment_matrix = np.zeros((num_scenes, num_summary_sentences))
	for (i,j) in pairs:
		if x[i,j].X == 1.0: alignment_matrix[i, j] = 1
		else: alignment_matrix[i, j] = 0
	
	for j in range(num_summary_sentences):
		summary_sentences['sent'+str(j)] = set()
		#sim_aligned = [x for x in filtered_sim[:, j] if alignment_matrix[x, j] == 1 and filtered_sim[x, j] > 0]
		#if len(sim_aligned) > 0:
		for i in range(num_scenes):
			if alignment_matrix[i, j] == 1 and filtered_sim[i, j] > 0:
				summary_sentences['sent'+str(j)].add('scene'+str(i))

				if 'scene'+str(i) not in script_scenes: script_scenes['scene'+str(i)] = set()
				script_scenes['scene'+str(i)].add('sent'+str(j))

	if not os.path.exists(os.path.join(output_dir, log_file, '')): os.makedirs(os.path.join(output_dir, log_file, ''))
	with open(os.path.join(output_dir, log_file, code+'.json'), 'w') as outfile:
		obj = {}
		obj['scene'] = {}
		obj['summary'] = {}
		for scene in script_scenes:
			obj['scene'][scene] = list(script_scenes[scene])
		for sent in summary_sentences:
			obj['summary'][sent] = list(summary_sentences[sent])
		obj['num_scenes'] = num_scenes
		obj['num_summary_sentences'] = num_summary_sentences
		json.dump(obj, outfile)

def get_scriptbase_alignment(output_dir, code, dataset_dir):
	sentence_scene = {}
	with open(dataset_dir + '/' + movie_code[code] + '/scriptbase/sceneSentenceMap.txt', 'r') as f:
		for line in f.readlines():
			if line.strip() != '':
				cols = line.strip().split('\t')
				sc_idx = int(cols[0])
				sent_idx = cols[1].replace('[', '').replace(']', '').split(',')
				for s in sent_idx:
					if s.strip() != '':
						sentence_scene[int(s.strip())] = sc_idx

	script_scenes = {}
	summary_sentences = {}
	with open(dataset_dir + '/' + movie_code[code] + '/scriptbase/0_1', 'r') as f:
		for line in f.readlines():
			if line.strip() != '':
				cols = line.strip().split('\t')
				if cols[1] == 'y':
					su_idx = int(cols[0].split('-')[0])
					sc_idx = sentence_scene[int(cols[0].split('-')[1])]

					if 'sent'+str(su_idx) not in summary_sentences: summary_sentences['sent'+str(su_idx)] = set()					
					summary_sentences['sent'+str(su_idx)].add('scene'+str(sc_idx))

					if 'scene'+str(sc_idx) not in script_scenes: script_scenes['scene'+str(sc_idx)] = set()
					script_scenes['scene'+str(sc_idx)].add('sent'+str(su_idx))

	if not os.path.exists(os.path.join(output_dir, 'scriptbase', '')): os.makedirs(os.path.join(output_dir, 'scriptbase', ''))
	with open(os.path.join(output_dir, 'scriptbase', code+'.json'), 'w') as outfile:
		obj = {}
		obj['scene'] = {}
		obj['summary'] = {}
		for scene in script_scenes:
			obj['scene'][scene] = list(script_scenes[scene])
		for sent in summary_sentences:
			obj['summary'][sent] = list(summary_sentences[sent])
		obj['num_scenes'] = len(script_scenes)
		obj['num_summary_sentences'] = len(summary_sentences)
		json.dump(obj, outfile)

def run_experiment(log_file, n, top_n=False, sim_matrix='bm25', limit_candidate=True, ensure_upward=True, ilp=True, coreference=True, compute_similarity=True, run_alignment=True):
	if not os.path.exists(dir_path + '/evaluation_results/'): os.makedirs(dir_path + '/evaluation_results/')
	with open(dir_path + '/evaluation_results/'+log_file+'.txt', 'w') as f:

		sim_matrix_dir = dir_path + '/data/sim_matrix/'
		output_dir = dir_path + '/output_files/'

		sum_ratio = 0
		for key in movie_list:

			print('\n===Getting alignment for', movie_code[key], '...')

			tuc = time.perf_counter()
			(script_scenes, summary_sentences, script_scenes_nostopword, summary_sentences_nostopword, script_entities, summary_entities) = preprocess(key, dataset_dir, coreference=coreference, entity_removal=False, lemmatized=False)

			ratio = len(script_scenes) / float(len(summary_sentences))
			#ratio = math.ceil(ratio)
			sum_ratio += ratio

			tic = time.perf_counter()
			print(f'Preprocess script and summary in {tic - tuc:0.4f} seconds.')

			if compute_similarity:

				if 'bm25' in sim_matrix:
					bm25_similarity = get_similarity_bm25(sim_matrix_dir, key, script_scenes_nostopword, summary_sentences_nostopword, script_entities, summary_entities)			
				if 'w2v' in sim_matrix:
					w2v_similarity = get_similarity_w2v(sim_matrix_dir, key, script_scenes_nostopword, summary_sentences_nostopword, script_entities, summary_entities)
				if 'bert' in sim_matrix:
					bert_similarity = get_similarity_bert(sim_matrix_dir, key, script_scenes, summary_sentences)
				if 'sts' in sim_matrix:
					sts_similarity = get_similarity_sts(sim_matrix_dir, key, script_scenes, summary_sentences)

				entity_similarity = get_similarity_entity(sim_matrix_dir, key, script_entities, summary_entities, len(script_scenes), len(summary_sentences))

				toc = time.perf_counter()
				print(f'Compute similarity matrices in {toc - tic:0.4f} seconds.')
			else:
				tic = time.perf_counter()

				if 'bm25' in sim_matrix:
					bm25_similarity = get_similarity_bm25(sim_matrix_dir, key, None, None, None, None, load=True)			
				if 'w2v' in sim_matrix:
					w2v_similarity = get_similarity_w2v(sim_matrix_dir, key, None, None, None, None, load=True)
				if 'bert' in sim_matrix:
					bert_similarity = get_similarity_bert(sim_matrix_dir, key, None, None, load=True)
				if 'sts' in sim_matrix:
					sts_similarity = get_similarity_sts(sim_matrix_dir, key, None, None, load=True)

				entity_similarity = get_similarity_entity(sim_matrix_dir, key, None, None, 0, 0, load=True)

				toc = time.perf_counter()
				print(f'Load similarity matrices in {toc - tic:0.4f} seconds.')

			if sim_matrix == 'bm25':
				similarity = bm25_similarity
			elif sim_matrix == 'w2v':
				similarity = w2v_similarity
			elif sim_matrix == 'bm25.w2v':
				similarity = np.multiply(bm25_similarity, np.add(w2v_similarity, entity_similarity))
			elif sim_matrix == 'bm25.sts':
				similarity = np.multiply(bm25_similarity, sts_similarity)
			elif sim_matrix == 'bm25.bert':
				similarity = np.multiply(bm25_similarity, bert_similarity)
			elif sim_matrix == 'bm25.w2v.sts':
				similarity= np.multiply(np.multiply(bm25_similarity, sts_similarity), np.add(w2v_similarity, entity_similarity))
			elif sim_matrix == 'bm25.bert.sts':
				similarity= np.multiply(np.multiply(bm25_similarity, sts_similarity), np.add(bert_similarity, entity_similarity))

			percentile = 50
			if run_alignment:	
				if ilp:
					get_ilp_alignment(output_dir, log_file, key, similarity, limit_candidate, percentile)
				else:
					if top_n:
						get_alignment(output_dir, log_file, key, similarity, n, ensure_upward, limit_candidate, percentile)
					else:
						get_alignment(output_dir, log_file, key, similarity, len(script_scenes), ensure_upward, limit_candidate, percentile)

				tac = time.perf_counter()
				print(f'Running alignment algorithm in {tac - toc:0.4f} seconds.')

			print('\n===Evaluate', movie_code[key], 'with', sim_matrix, '...')

			print('Scene-to-sentence ratio: %.1f' % ratio)
			(kappa, _, _) = ev.kappa_agreement_movie(key, gold_alignment_dir, output_dir+log_file+'/')
			(report, _, _) = ev.precision_recall_f1score_movie(key, gold_alignment_dir, output_dir+log_file+'/')
			print(report)

		print('\n===Overall performance across', len(movie_list), 'movies...')

		print('Avg scene-to-sentence ratio: %.1f' % (sum_ratio / float(len(movie_list))))
		(kappa_per_movie, micro_kappa, macro_kappa) = ev.kappa_agreement(movie_list, gold_alignment_dir, output_dir+log_file+'/')
		f.write(str(kappa_per_movie) + '\n')
		f.write('micro avg kappa:  %.3f\n' % micro_kappa)
		f.write('macro avg kappa:  %.3f\n' % macro_kappa)

		(score_per_movie, micro_report, macro_p, macro_r, macro_f1) = ev.precision_recall_f1score(movie_list, gold_alignment_dir, output_dir+log_file+'/')
		f.write(str(score_per_movie) + '\n')
		f.write('micro avg (p r f1): %s\n' % micro_report)
		f.write('macro avg (p r f1): %.3f %.3f %.3f\n' % (macro_p, macro_r, macro_f1))

		print('micro avg (p r f1): %s\n' % micro_report)
		print('macro avg (p r f1): %.3f %.3f %.3f\n' % (macro_p, macro_r, macro_f1))

def run_upperbound(log_file):
	if not os.path.exists(dir_path + '/evaluation_results/'): os.makedirs(dir_path + '/evaluation_results/')
	from sklearn.metrics import classification_report
	from sklearn.metrics import precision_recall_fscore_support

	precision_avg = 0
	recall_avg = 0
	f1score_avg = 0

	with open(dir_path + '/evaluation_results/'+log_file+'.txt', 'w') as f:
		for key in movie_list:
			f.write ('\nUpperbound for ' + key + '...\n')
			with open(gold_alignment_dir+'/'+key+'.json', 'r') as f1:
				data1 = json.load(f1)

				num_scenes = data1['num_scenes']
				num_summary_sentences = data1['num_summary_sentences']
				ratio = num_scenes / float(num_summary_sentences)
				ratio = math.ceil(ratio)

				alignment_matrix1 = np.zeros((num_scenes, num_summary_sentences))
				alignment_matrix2 = np.zeros((num_scenes, num_summary_sentences))

				scene1 = data1['scene']

				for scene in scene1:
					for sent in scene1[scene]:
						sc = int(scene.replace('scene', ''))
						se = int(sent.replace('sent', ''))
						alignment_matrix1[sc, se] = 1
						
				# diagonal line boundary
				for j in range(num_summary_sentences):
					for i in range(num_scenes):
						i_100 = (i/float(num_scenes)) * 100
						j_100 = (j/float(num_summary_sentences)) * 100

						if j_100 < i_100 + 30 and i_100 < j_100 + 30:
							alignment_matrix2[i, j] = alignment_matrix1[i, j]	
						else:
							alignment_matrix2[i, j] = 0

				# upper and lower diagonal is enforced
				prev_min_i = 0
				for j in range(num_summary_sentences):
					aligned = np.where(alignment_matrix1[:, j] == 1)[0]
					if aligned != []:
						min_i = aligned[0]
						if min_i < prev_min_i:
							for i in range(min_i, prev_min_i):
								alignment_matrix2[i, j] = 0
						else:
							prev_min_i = min_i

				prev_max_i = num_scenes
				for j in reversed(range(num_summary_sentences)):
					aligned = np.where(alignment_matrix1[:, j] == 1)[0]
					if aligned != []:
						max_i = aligned[-1]
						if max_i > prev_max_i:
							for i in range(prev_max_i+1, num_scenes):
								alignment_matrix2[i, j] = 0
						else:
							prev_max_i = max_i

				# max aligned with 5 scenes
				for j in range(num_summary_sentences):
					aligned = np.where(alignment_matrix2[:, j] == 1)[0]
					max = 5
					#if ratio > 5: max = ratio
					if len(aligned) > max:
						#print('!!!', aligned, max)
						for i in range(max, len(aligned)):
							alignment_matrix2[aligned[i], j] = 0

				
				for j in range(num_summary_sentences):
					aligned = np.where(alignment_matrix2[:, j] == 1)[0]
					max = 5
					#if ratio > 5: max = ratio
					if aligned != []:
						max_cont = aligned[0] + max
						if aligned[-1] > max_cont:
							#print('!!!', aligned, max_cont, aligned[-1])
							for i in range(max_cont+1, aligned[-1]+1):
								alignment_matrix2[i, j] = 0
				
				

				alignment1 = []
				for i in range(num_scenes):
					for j in range(num_summary_sentences):
						alignment1.append(int(alignment_matrix1[i, j]))

				alignment2 = []
				for i in range(num_scenes):
					for j in range(num_summary_sentences):
						alignment2.append(int(alignment_matrix2[i, j]))

				target_names = ['non-align', 'align']
				f.write (classification_report(alignment1, alignment2, target_names=target_names))
				f.write ('\n')

				(precision, recall, f1score, _) = precision_recall_fscore_support(alignment1, alignment2, average='macro')
				precision_avg += precision
				recall_avg += recall
				f1score_avg += f1score

		precision_avg = precision_avg / len(movie_code)
		recall_avg = recall_avg / len(movie_code)
		f1score_avg = f1score_avg / len(movie_code)

		f.write('macro avg (p r f1): %.3f %.3f %.3f\n' % (precision_avg, recall_avg, f1score_avg))


def run_baseline(log_file):
	if not os.path.exists(dir_path + '/evaluation_results/'): os.makedirs(dir_path + '/evaluation_results/')
	with open(dir_path + '/evaluation_results/'+log_file+'.txt', 'w') as f:
		output_dir = dir_path + '/output_files/'

		for key in movie_list:			
			print('\n===Getting alignment for', movie_code[key], '...')
			get_scriptbase_alignment(output_dir, key, dataset_dir)

			(kappa, _, _) = ev.kappa_agreement_movie(key, gold_alignment_dir, output_dir+log_file+'/')
			(report, _, _) = ev.precision_recall_f1score_movie(key, gold_alignment_dir, output_dir+log_file+'/')
			print(report)
			
		print('\n===Overall performance across', len(movie_list), 'movies...')

		print('Avg scene-to-sentence ratio: %.1f' % (sum_ratio / float(len(movie_list))))

		(kappa_per_movie, micro_kappa, macro_kappa) = ev.kappa_agreement(movie_code, gold_alignment_dir, output_dir+log_file+'/')
		f.write(str(kappa_per_movie) + '\n')
		f.write('micro avg kappa:  %.3f\n' % micro_kappa)
		f.write('macro avg kappa:  %.3f\n' % macro_kappa)

		(score_per_movie, micro_report, macro_p, macro_r, macro_f1) = ev.precision_recall_f1score(movie_code, gold_alignment_dir, output_dir+log_file+'/')
		f.write(str(score_per_movie) + '\n')
		f.write('micro avg (p r f1): %s\n' % micro_report)
		f.write('macro avg (p r f1): %.3f %.3f %.3f\n' % (macro_p, macro_r, macro_f1))

		print('micro avg (p r f1): %s\n' % micro_report)
		print('macro avg (p r f1): %.3f %.3f %.3f\n' % (macro_p, macro_r, macro_f1))

def run_scriptbase(k, dataset_dir, n, top_n=False, sim_matrix='bm25.w2v.sts', limit_candidate=True, ensure_upward=True, ilp=True, coreference=True, compute_similarity=True, run_alignment=True):
	#movies = os.listdir(dataset_dir)
	#key = sorted(movies)[k]

	#sorted list of movies based on compression ratio of scenes-to-summary
	movies = []
	with open('scriptbase_statistic_sorted.csv', 'r', newline='') as csvfile:
		reader = csv.reader(csvfile)
		next(reader, None)  # skip the headers
		for row in reader: movies.append(row[0])
	key = movies[k]

	eprint('\n=== ', k, ' Getting alignment for', key, '...')

	sim_matrix_dir = os.path.join(dataset_dir, key, 'alignarr_similarity', '')

	if compute_similarity:
		tuc = time.perf_counter()
		(script_scenes, summary_sentences, script_scenes_nostopword, summary_sentences_nostopword, script_entities, summary_entities) = preprocess_scriptbase(key, dataset_dir, coreference=coreference, entity_removal=False, lemmatized=False)

		ratio = len(script_scenes) / float(len(summary_sentences))
		#ratio = math.ceil(ratio)

		tic = time.perf_counter()
		eprint(f'   Preprocess script and summary in {tic - tuc:0.4f} seconds.')

		bm25_similarity = get_similarity_bm25(sim_matrix_dir, '', script_scenes_nostopword, summary_sentences_nostopword, script_entities, summary_entities, load=False)			
		w2v_similarity = get_similarity_w2v(sim_matrix_dir, '', script_scenes_nostopword, summary_sentences_nostopword, script_entities, summary_entities, load=False)
		sts_similarity = get_similarity_sts(sim_matrix_dir, '', script_scenes, summary_sentences, load=False)
		entity_similarity = get_similarity_entity(sim_matrix_dir, '', script_entities, summary_entities, len(script_scenes), len(summary_sentences), load=False)
		
		toc = time.perf_counter()
		eprint(f'   Compute similarity matrices in {toc - tic:0.4f} seconds.')
	else:
		tic = time.perf_counter()

		bm25_similarity = get_similarity_bm25(sim_matrix_dir, '', None, None, None, None, load=True)			
		w2v_similarity = get_similarity_w2v(sim_matrix_dir, '', None, None, None, None, load=True)
		sts_similarity = get_similarity_sts(sim_matrix_dir, '', None, None, load=True)
		entity_similarity = get_similarity_entity(sim_matrix_dir, '', None, None, 0, 0, load=True)
		
		toc = time.perf_counter()
		eprint(f'   Load similarity matrices in {toc - tic:0.4f} seconds.')

	similarity = np.multiply(bm25_similarity, sts_similarity)

	percentile = 50
	if run_alignment:	
		if ilp:
			get_ilp_alignment(dataset_dir, key, 'alignarr_alignment', similarity, limit_candidate, percentile)
		else:
			if top_n:
				get_alignment(dataset_dir, key, 'alignarr_alignment', similarity, n, ensure_upward, limit_candidate, percentile)
			else:
				get_alignment(dataset_dir, key, 'alignarr_alignment', similarity, len(script_scenes), ensure_upward, limit_candidate, percentile)

		tac = time.perf_counter()
		eprint(f'Running alignment algorithm in {tac - toc:0.4f} seconds.')


if __name__ == "__main__":  
	if len(sys.argv) > 3 and sys.argv[1] == 'scriptbase':
		dataset_dir = sys.argv[2]
		k = int(sys.argv[3])
		if len(sys.argv) > 4 and sys.argv[4] == 'sim_matrix':
			run_scriptbase(k, dataset_dir, 1, top_n=False, sim_matrix='bm25.sts', compute_similarity=True, run_alignment=False)
		else:
			run_scriptbase(k, dataset_dir, 1, top_n=False, sim_matrix='bm25.sts', compute_similarity=False, run_alignment=True)

	elif len(sys.argv) > 1:
		k = int(sys.argv[1])
		key = movie_list[k]
		print('\n===Debugging', key, '...')

		(script_scenes, summary_sentences, script_scenes_nostopword, summary_sentences_nostopword, script_entities, summary_entities) = preprocess(key, dataset_dir, coreference=True, entity_removal=False, lemmatized=False)

		ratio = len(script_scenes) / float(len(summary_sentences))
		#ratio = math.ceil(ratio)

		sim_matrix_dir = dir_path + '/data/sim_matrix/'
		output_dir = dir_path + '/output_files/'

		if len(sys.argv) > 2:
			sim_matrix = sys.argv[2]
			
			if sim_matrix == 'bm25':
				tic = time.perf_counter()	
				bm25_similarity = get_similarity_bm25(sim_matrix_dir, key, script_scenes_nostopword, summary_sentences_nostopword, script_entities, summary_entities, load=False)
				toc = time.perf_counter()
				print(f'Getting {sim_matrix} similarity matrix for {key} in {toc - tic:0.4f} seconds.')
				
			elif sim_matrix == 'w2v':
				tic = time.perf_counter()
				w2v_similarity = get_similarity_w2v(sim_matrix_dir, key, script_scenes_nostopword, summary_sentences_nostopword, script_entities, summary_entities, load=False)
				toc = time.perf_counter()
				print(f'Getting {sim_matrix} similarity matrix for {key} in {toc - tic:0.4f} seconds.')

				entity_similarity = get_similarity_entity(sim_matrix_dir, key, script_entities, summary_entities, len(script_scenes), len(summary_sentences), load=False)
				tac = time.perf_counter()
				print(f'Getting entity similarity matrix for {key} in {tac - toc:0.4f} seconds.')

			elif sim_matrix == 'bert':
				tic = time.perf_counter()
				bert_similarity = get_similarity_bert(sim_matrix_dir, key, script_scenes, summary_sentences, load=False)
				toc = time.perf_counter()
				print(f'Getting {sim_matrix} similarity matrix for {key} in {toc - tic:0.4f} seconds.')

			elif sim_matrix == 'sts':	#sts similarity requires w2v and entity (word overlap) similarity
				tic = time.perf_counter()
				w2v_similarity = get_similarity_w2v(sim_matrix_dir, key, script_scenes_nostopword, summary_sentences_nostopword, script_entities, summary_entities, load=False)
				toc = time.perf_counter()
				print(f'Getting w2v similarity matrix for {key} in {toc - tic:0.4f} seconds.')

				entity_similarity = get_similarity_entity(sim_matrix_dir, key, script_entities, summary_entities, len(script_scenes), len(summary_sentences), load=False)
				tac = time.perf_counter()
				print(f'Getting entity similarity matrix for {key} in {tac - toc:0.4f} seconds.')

				sts_similarity = get_similarity_sts(sim_matrix_dir, key, script_scenes, summary_sentences, load=False)
				tuc = time.perf_counter()
				print(f'Getting {sim_matrix} similarity matrix for {key} in {tuc - tac:0.4f} seconds.')

			elif sim_matrix == 'sts.nli':
				(sts_similarity, nli_similarity) = get_similarity_sts_nli(sim_matrix_dir, key, script_scenes, summary_sentences, load=False)

			elif sim_matrix == 'bertscore':
				bert_score_similarity = get_similarity_bert_score(sim_matrix_dir, key, script_scenes, summary_sentences, load=False)

		else:
			bm25_similarity = get_similarity_bm25(sim_matrix_dir, key, script_scenes_nostopword, summary_sentences_nostopword, script_entities, summary_entities, load=True)
			w2v_similarity = get_similarity_w2v(sim_matrix_dir, key, script_scenes_nostopword, summary_sentences_nostopword, script_entities, summary_entities, load=True)
			(sts_similarity, nli_similarity) = get_similarity_sts_nli(sim_matrix_dir, key, script_scenes, summary_sentences, load=True)
			entity_similarity = get_similarity_entity(sim_matrix_dir, key, script_entities, summary_entities, len(script_scenes), len(summary_sentences), load=True)
			
			similarity = np.multiply(np.multiply(bm25_similarity, sts_similarity), np.add(w2v_similarity, entity_similarity))
			
			get_ilp_alignment(output_dir, 'auto_alignment_'+key, key, similarity)

			(kappa, _, _) = ev.kappa_agreement_movie(key, gold_alignment_dir, dir_path + '/output_files/auto_alignment_'+key+'/')
			print('kappa', kappa)

			(report, _, _) = ev.precision_recall_f1score_movie(key, gold_alignment_dir, dir_path + '/output_files/auto_alignment_'+key+'/')
			print(report)
			
		
	else:
		#ACL 2021
		print('Experiments...')

		#Baseline
		run_baseline('scriptbase')
		
		#Ablation
		run_experiment('auto_alignment_bm25', 1, top_n=False, sim_matrix='bm25', compute_similarity=False, run_alignment=True)
		#run_experiment('auto_alignment_w2v', 1, top_n=False, sim_matrix='w2v', compute_similarity=False, run_alignment=True)
		run_experiment('auto_alignment_bm25.w2v', 1, top_n=False, sim_matrix='bm25.w2v', compute_similarity=False, run_alignment=True)
		run_experiment('auto_alignment_bm25.w2v.sts', 1, top_n=False, sim_matrix='bm25.w2v.sts', compute_similarity=False, run_alignment=True)
		run_experiment('auto_alignment_bm25.sts', 1, top_n=False, sim_matrix='bm25.sts', compute_similarity=False, run_alignment=True)

		run_experiment('auto_alignment_bm25.w2v.sts_non_ilp', 1, top_n=False, sim_matrix='bm25.w2v.sts', ilp=False, compute_similarity=False, run_alignment=True)
		run_experiment('auto_alignment_bm25.sts_non_ilp', 1, top_n=False, sim_matrix='bm25.sts', ilp=False, compute_similarity=False, run_alignment=True)

		run_experiment('auto_alignment_bm25.bert.sts', 1, top_n=False, sim_matrix='bm25.bert.sts', compute_similarity=False, run_alignment=True)
		run_experiment('auto_alignment_bm25.bert', 1, top_n=False, sim_matrix='bm25.bert', compute_similarity=False, run_alignment=True)

		run_upperbound('upperbound')
