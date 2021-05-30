import sys
import json
import os

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

def run(film):
	print('processing'+film+'...')
	with open(film+'.txt', 'r') as f:
		for line in f.readlines():
			if not line.startswith('#'):
				#print(line.strip())

				cols = line.strip().split(': ')
				summary = 'sent'+cols[0]

				if len(cols) > 1:
					scenes = cols[1].split(',')

					jsonfile = film+'.json'
					if not os.path.exists(jsonfile):
						with open(jsonfile, 'w') as outfile:
							obj = {}
							if 'scene' not in obj: obj['scene'] = {}
							if 'summary' not in obj: obj['summary'] = {}    
							json.dump(obj, outfile)

					with open(jsonfile, 'r') as infile:
						obj = json.load(infile)
						if 'scene' not in obj: obj['scene'] = {}
						if 'summary' not in obj: obj['summary'] = {}

						for id in scenes:
							scene = 'scene'+id.strip()
							if scene in obj['scene']:
								if summary not in obj['scene'][scene]: obj['scene'][scene].append(summary)
							else:
								obj['scene'][scene] = []
								obj['scene'][scene].append(summary)
							if summary in obj['summary']:
								if scene not in obj['summary'][summary]: obj['summary'][summary].append(scene)
							else:
								obj['summary'][summary] = []
								obj['summary'][summary].append(scene)

					with open(jsonfile, 'w') as outfile:
						json.dump(obj, outfile)	

if len(sys.argv) > 1:
	run(sys.argv[1])
else:
	for key in movie_code:
		run(key)				