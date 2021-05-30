import sys
import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

def kappa_agreement_movie(code, alignment_dir1, alignment_dir2):
	with open(alignment_dir1+'/'+code+'.json', 'r') as f1:
		with open(alignment_dir2+'/'+code+'.json', 'r') as f2:
			data1 = json.load(f1)
			data2 = json.load(f2)

			num_scenes = data1['num_scenes']
			num_summary_sentences = data1['num_summary_sentences']

			alignment_matrix1 = np.zeros((num_scenes, num_summary_sentences))
			alignment_matrix2 = np.zeros((num_scenes, num_summary_sentences))

			scene1 = data1['scene']
			scene2 = data2['scene']

			for scene in scene1:
				for sent in scene1[scene]:
					sc = int(scene.replace('scene', ''))
					se = int(sent.replace('sent', ''))
					alignment_matrix1[sc, se] = 1

			for scene in scene2:
				for sent in scene2[scene]:
					sc = int(scene.replace('scene', ''))
					se = int(sent.replace('sent', ''))
					alignment_matrix2[sc, se] = 1

			alignment1 = []
			for i in range(num_scenes):
				for j in range(num_summary_sentences):
					alignment1.append(alignment_matrix1[i, j])

			alignment2 = []
			for i in range(num_scenes):
				for j in range(num_summary_sentences):
					alignment2.append(alignment_matrix2[i, j])

			kappa = cohen_kappa_score(alignment1, alignment2)
			return (kappa, alignment1, alignment2)	

def kappa_agreement(movie_code, alignment_dir1, alignment_dir2):
	macro_kappa = 0
	kappa_per_movie = {}

	alignment1_all = []
	alignment2_all = []

	for i, key in enumerate(movie_code):
		(kappa, alignment1, alignment2) = kappa_agreement_movie(key, alignment_dir1, alignment_dir2)
		macro_kappa += kappa

		kappa_per_movie[key] = kappa
		
		alignment1_all.extend(alignment1)
		alignment2_all.extend(alignment2)

	micro_kappa = cohen_kappa_score(alignment1_all, alignment2_all)
	macro_kappa = macro_kappa / len(movie_code)

	return (kappa_per_movie, micro_kappa, macro_kappa)

def precision_recall_f1score_movie(code, alignment_dir1, alignment_dir2):
	with open(alignment_dir1+'/'+code+'.json', 'r') as f1:
		with open(alignment_dir2+'/'+code+'.json', 'r') as f2:
			data1 = json.load(f1)
			data2 = json.load(f2)

			num_scenes = data1['num_scenes']
			num_summary_sentences = data1['num_summary_sentences']

			alignment_matrix1 = np.zeros((num_scenes, num_summary_sentences))
			alignment_matrix2 = np.zeros((num_scenes, num_summary_sentences))

			scene1 = data1['scene']
			scene2 = data2['scene']

			for scene in scene1:
				for sent in scene1[scene]:
					sc = int(scene.replace('scene', ''))
					se = int(sent.replace('sent', ''))
					alignment_matrix1[sc, se] = 1

			for scene in scene2:
				for sent in scene2[scene]:
					sc = int(scene.replace('scene', ''))
					se = int(sent.replace('sent', ''))
					alignment_matrix2[sc, se] = 1

			alignment1 = []
			for i in range(num_scenes):
				for j in range(num_summary_sentences):
					alignment1.append(int(alignment_matrix1[i, j]))

			alignment2 = []
			for i in range(num_scenes):
				for j in range(num_summary_sentences):
					alignment2.append(int(alignment_matrix2[i, j]))

			target_names = ['non-align', 'align']
			return (classification_report(alignment1, alignment2, target_names=target_names), alignment1, alignment2)

def precision_recall_f1score(movie_code, alignment_dir1, alignment_dir2):
	macro_p = 0
	macro_r = 0
	macro_f1 = 0
	
	score_per_movie = ''

	alignment1_all = []
	alignment2_all = []

	for i, key in enumerate(movie_code):
		(report, alignment1, alignment2) = precision_recall_f1score_movie(key, alignment_dir1, alignment_dir2)
		score_for_macro = report.split('\n')[6]
		score_for_macro = score_for_macro.replace(' macro avg', key).strip()
		scores = re.split('\s+', score_for_macro)

		score_per_movie += '\n' + key + '\n'
		score_per_movie += report

		precision = float(scores[1])
		recall = float(scores[2])
		f1 = float(scores[3])

		macro_p += precision
		macro_r += recall
		macro_f1 += f1
		
		alignment1_all.extend(alignment1)
		alignment2_all.extend(alignment2)

	target_names = ['non-align', 'align']
	micro_report = classification_report(alignment1_all, alignment2_all, target_names=target_names)
	
	macro_p = macro_p / len(movie_code)
	macro_r = macro_r / len(movie_code)
	macro_f1 = macro_f1 / len(movie_code)

	return (score_per_movie, micro_report, macro_p, macro_r, macro_f1)