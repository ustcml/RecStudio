================================
recstudio.model.basemodel
================================

The models are divided into 4 basic classes according to the number of towers(encoders).
In Recommender Systems, classical Factorized Machine based model treated user, item and other information
as data features, so there is no encoder in those methods. Besides, auto encoder based methods encoder items
and get user representations with its interacted items. Typical matrix factorized methods encode both users
and items, then calculating the scores with certain score function, such as inner product, multi layer perceptron.



- :doc:`recstudio.model.basemodel.Recommender <base>`
- :doc:`recstudio.model.basemodel.TowerFreeRecommender <towerfree>`
- :doc:`recstudio.model.basemodel.ItemTowerRecommender <itemtower>`
- :doc:`recstudio.model.basemodel.TwoTowerRecommender <twotower>`