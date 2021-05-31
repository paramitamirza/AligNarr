# AligNarr: Aligning Narratives on Movies

### Requirements

* [AllenNLP Coreference Resolution](https://demo.allennlp.org/coreference-resolution)
* [Truecaser](https://github.com/nreimers/truecaser)
* [FuzzyWuzzy string matching](https://github.com/seatgeek/fuzzywuzzy)
* [Gurobi Optimizer](https://www.gurobi.com/products/gurobi-optimizer/) -- [license](https://www.gurobi.com/academia/academic-program-and-licenses/) should be installed on the machine

#### Install python environment
* `conda env create -f env/environment.yml`, or
* `pip install -r env/requirements.txt`

### ACL 2021 Experiments

* Run `python alignarr.py` 

#### Publication
Paramita Mirza, Mostafa Abouhamra and Gerhard Weikum (2021). AligNarr: Aligning Narratives on Movies. *In Proceedings of ACL 2021.* [[pdf]](https://d5demos.mpi-inf.mpg.de/alignarr/static/575_file_Paper.pdf)
