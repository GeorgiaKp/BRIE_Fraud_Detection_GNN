This is the code implementation of the paper "Enhancing Fraud Detection in GNNs with
Synthetic Fraud Generation and Integrated Structural Features" (ICANN 2024).

## Dataset

YelpChi and Amazon can be downloaded from [here](https://github.com/YingtongDou/CARE-GNN/tree/master/data).

Run inside the data directory `wget https://github.com/YingtongDou/CARE-GNN/raw/master/data/YelpChi.zip && unzip YelpChi.zip` 

Run `python src/data_process.py` to pre-process the data.


## Usage

```sh
python main.py --config ./config/brie_yelpchi.yml
```

## Brief Description

### Introduction
<img src="images/intro.jpg" alt="Intro" width="709" height="595"/>

### Contribution - BRIE
<img src="images/contribution.jpg" alt="contribution" width="693" height="295"/>

#### Additional structural features (Contribution 1)
<img src="images/additional_features.jpg" alt="additional" width="692" height="672"/>

#### Synthesizing fraud nodes (Contribution 2)
<img src="images/synthesizing.jpg" alt="synthesizing" width="626" height="357"/>

#### Results
<img src="images/results.jpg" alt="results" width="621" height="320"/>

#### Conclusion
<img src="images/conclusion.jpg" alt="conclusion" width="700" height="384"/>