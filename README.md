# NHOD: A High-dimensional Outlier Detection Approach Based on Local Coulomb Force

A High-dimensional Outlier Detection Approach Based on Local Coulomb Force.

### Abstract
Traditional outlier detections are inadequate for high-dimensional data analysis due to the interference of distance tending to be concentrated (“curse of dimensionality”). Inspired by the Coulomb's law, we propose a new high-dimensional data similarity measure vector, which consists of outlier Coulomb force and outlier Coulomb resultant force. Outlier Coulomb force not only effectively gauges similarity measures among data objects, but also fully reflects differences among dimensions of data objects by vector projection in each dimension. More importantly, Coulomb resultant force can effectively measure deviations of data objects from a data center, making detection results interpretable. We introduce a new neighborhood outlier factor, which drives the development of a high-dimensional outlier detection algorithm. In our approach, attribute values with a high deviation degree is treated as interpretable information of outlier data. Finally, we implement and evaluate our algorithm using the UCI and synthetic datasets. Our experimental results show that the algorithm effectively alleviates the interference of “Curse of Dimensionality”. The findings confirm that high-dimensional outlier data originated by the algorithm are interpretable.

### How to use
```
python NHOD.py
```

### Experiment
All experimental data are sourced from [UCI](https://archive.ics.uci.edu/), [ODDS](https://odds.cs.stonybrook.edu/), and [DAMI](https://www.dbs.ifi.lmu.de/research/outlier-evaluation/DAMI/).

We are positioned to evaluate the detection accuracy of NHOD, which is compared against the six mainstream outlier detection algorithms handling the six UCI datasets. The experimental and comparison results are tabulated in Table 1.



|Algorithm\Dateset|Mnist|Musk|PInternetAds|KDDCup99|Arrhythmia|HAPT|
|----|----|----|----|----|----|----|
||k=88 |k=55|k=57|k=247 | k=21 |  k=105|
|**AUC**| | | | | | |
|**NHOD**|**88.27**|**100.0**|**72.18**|**98.93**|**81.33**|**97.21**|
|LGOD	|63.08	|70.95|67.89	|81.33	|69.96	|80.09|
|LOF	|79.24	|35.47	|62.60	|68.84	|75.92	|53.25|
|RDOS	|74.25	|44.12	|71.78	|86.16	|76.41	|66.13|
|MOD+	|84.13	|96.25	|59.86	|98.42	|79.70	|94.92|
|ABOD	|36.53	|3.447	|53.77	|75.44	|76.84	|92.73|
|SOD	|55.21	|88.16	|59.49	|94.81	|71.20	|89.97|
|**ACC**| | | | | | |
|**NHOD**	|**89.37**|**100.0**|**86.55**|99.51|**84.96**|**95.70**|
|LGOD	|85.95	|95.56	|81.07	|99.40	|81.42	|93.41|
|LOF	|88.19	|93.86	|80.30	|99.19	|83.19  |90.92|
|RDOS	|87.90	|94.12	|80.82	|**99.57**	|83.19	|92.19|
|MOD+	|88.93	|95.89	|78.92	|99.47	|84.96	|95.17|
|ABOD	|82.06	|93.66	|76.35	|99.60  |77.53	|94.22|
|SOD	|85.03	|94.58	|79.53	|99.28	|80.53	|93.38|


### How to Cite
Please cite this model using this format.

```
@article{zhu2022high,
  title={A high-dimensional outlier detection approach based on local coulomb force},
  author={Zhu, Pengyun and Zhang, Chaowei and Li, Xiaofeng and Zhang, Jifu and Qin, Xiao},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2022},
  publisher={IEEE}
}
```
