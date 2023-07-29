# Kepler Mission: Celestial Bodies Classification

![Kepler Mission](https://www.nasa.gov/sites/default/files/styles/card_page_banner/public/thumbnails/image/kepler-k2-web-banner2.0-nasa-9.jpeg)

This repository contains the code and analysis for the Celestial Bodies Classification project. The project focuses on processing data from the NASA Kepler mission to classify observed celestial objects as Exoplanets or not, using data mining algorithms and machine learning techniques.

## Abstract
In Astronomy, the study of celestial bodies relies on data collected from satellites and space telescopes. The data from the NASA Kepler mission, in particular, offers valuable insights into Exoplanets (planets outside our solar system). With the vast volume of data collected, traditional manual analysis becomes impractical. Therefore, this project aims to efficiently analyze Big Data in Astrophysics using Data Mining algorithms and Machine Learning techniques to classify observed objects from the Kepler orbital telescope as Exoplanets or not.

## Keywords
Kepler, Exoplanets, Classification, Predictive modeling

## Introduction
### Application Domain and Research Problem
The Kepler space telescope, launched by NASA in 2009 and active until 2014, was designed to gather crucial data for detecting exoplanets orbiting stars in various regions of the Milky Way. The mission used photometric observations of stars to detect transits of planets in front of their host stars, leading to numerous exoplanet discoveries. This project utilizes data from the Kepler mission to classify observed objects accurately as Exoplanets or not using data mining techniques.

## Related Previous Work
For this project, relevant machine learning techniques applied in exoplanet detection were consulted. The analysis of previous work aided in understanding the complexity of the task and informed the selection of appropriate methods and models.

## Data Preprocessing and Exploration
The code in `pre-processing.py` contains the data preprocessing steps, including feature selection, handling missing values, and target variable transformation. After preprocessing, the data is saved as `data_preprocessed.csv`.

The code in `EDA.py` performs exploratory data analysis, visualizing various features to gain insights into the data distribution.

## Modeling
The code in `modelling.py` contains the implementation of four models for celestial bodies classification: Logistic Regression, Decision Tree, Random Forest, and Neural Network. The dataset is split into training and test sets, and the features are scaled before training the models. Evaluation metrics such as precision, recall, F1-score, accuracy, and area under the ROC curve (AUC) are calculated for each model.

### Result Insights
Here are the evaluation metrics for each model:

- Logistic Regression:
  - Precision: 0.6652935118434603
  - Recall: 0.8579017264276229
  - F1 Score: 0.7494199535962878
  - Accuracy: 0.8363016294050777
  - AUC: 0.9090171656956479

- Decision Tree:
  - Precision: 0.6789087093389297
  - Recall: 0.8592297476759628
  - F1 Score: 0.7584994138335286
  - Accuracy: 0.8438802576733612

- Random Forest:
  - Precision: 0.6635802469135802
  - Recall: 0.8565737051792829
  - F1 Score: 0.7478260869565218
  - Accuracy: 0.8351648351648352

- Neural Network:
  - Precision: 0.6631908237747653
  - Recall: 0.8446215139442231
  - F1 Score: 0.7429906542056075
  - Accuracy: 0.8332701780977643

## Conclusion
This project demonstrates the effective use of Data Mining techniques and Machine Learning algorithms in classifying celestial bodies observed by the NASA Kepler mission. The Random Forest model shows promising results for this classification task.

## Further Developments
The project can be expanded by exploring more complex models, such as Multi-Layers Neural Networks, to potentially improve classification performance. Additionally, a greater k value can be utilized for k-fold Cross Validation to enhance model evaluation.

## References
1. [Exoplanets 101 - NASA](https://exoplanets.nasa.gov/the-search-for-life/exoplanets-101/)
2. [Kepler Mission Overview](https://www.nasa.gov/mission_pages/kepler/overview/index.html)
3. [Identifying Exoplanets with Deep Learning: A Five-planet Resonant Chain around Kepler-80 and an Eighth Planet around Kepler-90 - Christopher J. Shallue, Andrew Vanderburg](https://lweb.cfa.harvard.edu/~avanderb/kepler90i.pdf)
4. [Kaggle - Kepler Exoplanet Search Results](https://www.kaggle.com/nasa/kepler-exoplanet-search-results)
5. [Kepler Candidate Columns - Exoplanet Archive, IPAC, Caltech](https://exoplanetarchive.ipac.caltech.edu/docs/API_kepcandidate_columns.html)

For any questions or inquiries, please contact Kanish Chugh at kanishchugh2001@gmail.com
