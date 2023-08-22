# NHOD: A High-dimensional Outlier Detection Approach Based on Local Coulomb Force

A High-dimensional Outlier Detection Approach Based on Local Coulomb Force.

### Abstract
Traditional outlier detections are inadequate for high-dimensional data analysis due to the interference of distance tending to be concentrated (“curse of dimensionality”). Inspired by the Coulomb's law, we propose a new high-dimensional data similarity measure vector, which consists of outlier Coulomb force and outlier Coulomb resultant force. Outlier Coulomb force not only effectively gauges similarity measures among data objects, but also fully reflects differences among dimensions of data objects by vector projection in each dimension. More importantly, Coulomb resultant force can effectively measure deviations of data objects from a data center, making detection results interpretable. We introduce a new neighborhood outlier factor, which drives the development of a high-dimensional outlier detection algorithm. In our approach, attribute values with a high deviation degree is treated as interpretable information of outlier data. Finally, we implement and evaluate our algorithm using the UCI and synthetic datasets. Our experimental results show that the algorithm effectively alleviates the interference of “Curse of Dimensionality”. The findings confirm that high-dimensional outlier data originated by the algorithm are interpretable.

### How to use
```
python NHOD.py
```

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
