import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score

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

def get_alignment_plot(alignment_dir, code):
	with open(alignment_dir+'/'+code+'.json', 'r') as f:
		data = json.load(f)

		# x as scenes and y as summary sentences
		x_list = []
		y_list = []
		for sc in data['scene']:
			sc_idx = int(sc.replace('scene', ''))
			for su in data['scene'][sc]:
				su_idx = int(su.replace('sent', ''))
				x_list.append((sc_idx/float(data['num_scenes'])*100))
				y_list.append((su_idx/float(data['num_summary_sentences'])*100))

		x = np.array(x_list)
		y = np.array(y_list)

		return (x, y)

def plot_alignment(alignment_dir, alignment_name):
	fig = plt.figure(figsize=(11,3), frameon=False)
	
	ax1 = fig.add_subplot(111)
	ax1.set_ylabel('summary')
	ax1.set_xlabel('script')
	ax1.tick_params(axis=u'both', which=u'both',length=0)
	ax1.set_yticklabels([])
	ax1.set_xticklabels([])
	ax1.spines['top'].set_visible(False)
	ax1.spines['right'].set_visible(False)
	ax1.spines['bottom'].set_visible(False)
	ax1.spines['left'].set_visible(False)
	
	colors = ['xkcd:sky blue', 'xkcd:teal', 'xkcd:grey', 'xkcd:tan', 'xkcd:pink', 'xkcd:mauve', 'xkcd:beige', 'xkcd:turquoise', 'xkcd:mustard', 'xkcd:lavender']

	for i, key in enumerate(movie_code):
		(x, y) = get_alignment_plot(alignment_dir, key)
		ax1.scatter(x, y, s=10, c=colors[i], marker="s", label=movie_code[key])
	
	ax1.plot(x, x + 30, linestyle='solid', color='xkcd:blue', linewidth=0.5)
	ax1.plot(x + 30, x, linestyle='solid', color='xkcd:blue', linewidth=0.5)

	box = ax1.get_position()
	ax1.set_position([box.x0, box.y0, box.width * 0.65, box.height])
	ax1.legend(loc='center left', prop={'size':11}, bbox_to_anchor=(1, 0.5))
	
	#plt.show()
	plt.axis([-1, 101, -1, 101])
	plt.savefig(alignment_name+'.pdf', bbox_inches='tight')

def get_overlapped_alignment(code, alignment_dir1, alignment_dir2):
	script_scenes = {}
	summary_sentences = {}

	with open(alignment_dir1+'/'+code+'.json', 'r') as f1:
		with open(alignment_dir2+'/'+code+'.json', 'r') as f2:
			data1 = json.load(f1)
			data2 = json.load(f2)

			num_scenes = data1['num_scenes']
			num_summary_sentences = data1['num_summary_sentences']

			scene1 = data1['scene']
			scene2 = data2['scene']
			scene_both = set(scene1.keys()).union(set(scene2.keys()))

			for scene in scene_both:
				if scene in scene1 and scene in scene2:
					overlap = set(scene1[scene]).intersection(set(scene2[scene]))
					script_scenes[scene] = list(overlap)

			summary1 = data1['summary']
			summary2 = data2['summary']
			summary_both = set(summary1.keys()).union(set(summary2.keys()))

			for sent in summary_both:
				if sent in summary1 and sent in summary2:
					overlap = set(summary1[sent]).intersection(set(summary2[sent]))
					summary_sentences[sent] = list(overlap)
		
	if not os.path.exists('./output_files/overlapped'): os.makedirs('./output_files/overlapped/')
	with open('./output_files/overlapped/'+code+'.json', 'w') as outfile:
		obj = {}
		if 'scene' not in obj: obj['scene'] = script_scenes
		if 'summary' not in obj: obj['summary'] = summary_sentences  
		obj['num_scenes'] = num_scenes
		obj['num_summary_sentences'] = num_summary_sentences
		json.dump(obj, outfile)

def compute_agreement(code, alignment_dir1, alignment_dir2):
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

if __name__ == "__main__":
	if len(sys.argv) > 1:
		get_overlapped_alignment(sys.argv[1], './alignment1/', './alignment2/')

	else:
		plot_alignment('./alignment1/', 'alignment1')
		plot_alignment('./alignment2/', 'alignment2')

		macro_kappa = 0
		alignment1_all = []
		alignment2_all = []

		for i, key in enumerate(movie_code):
			get_overlapped_alignment(key, './alignment1/', './alignment2/')
			(kappa, alignment1, alignment2) = compute_agreement(key, './alignment1/', './alignment2/')
			macro_kappa += kappa

			print(key, 'kappa', kappa)
			
			alignment1_all.extend(alignment1)
			alignment2_all.extend(alignment2)

		micro_kappa = cohen_kappa_score(alignment1_all, alignment2_all)
		macro_kappa = macro_kappa / len(movie_code)

		print('-------')
		print('micro avg kappa', micro_kappa)
		print('macro avg kappa', macro_kappa)

		plot_alignment('./output_files/overlapped/', 'overlapped_alignment')

		run_name = 'auto_alignment_bm25_w2v_topn_limit_upward'
		plot_alignment('./output_files/'+run_name+'/', run_name)

		