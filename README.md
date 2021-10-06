# Solution-for-HackerEarth-Machine-Learning-challenge (26st place)
HackerEarth Machine Learning challenge: Of Genomes And Genetics [link](https://www.hackerearth.com/challenges/competitive/hackerearth-machine-learning-challenge-genetic-testing/).
# Why I took part in this competition?
Because: 
- I have wanted to practice some feature engineering techniques about tabular data, ensemble,... that I found on Kaggle [link](https://www.kaggle.com/vbmokin/data-science-for-tabular-data-advanced-techniques).
- I have wanted to take part in a competition which was ongoing. That has made me feel like I'm taking part in the Olympics in which a lot of competitors have to compete against each other.
# Time to complete 
That time when I found this [competition](https://www.hackerearth.com/challenges/competitive/hackerearth-machine-learning-challenge-genetic-testing/), It still had 1 week to close. So my work still had concentrated on [EDA](https://www.kaggle.com/thnhtonvng/eda-hacker-earthcompe/notebook) and Feature engineering. In this post, I will share how I did feature engineering.
# Train and Test
When performing Label Encoding below, you must encode train and test together ([reference from](https://www.kaggle.com/c/ieee-fraud-detection/discussion/108575))
# Feature Engineering Techniques
## Label Encode ( Categorical features )
- Features have 2 values:
  - Genes in mother's side
  - Birth defects
  - History of anomalies in previous pregnancie
  - Assisted conception IVF/ART
  - H/O serious maternal illness 
  - Folic acid details (peri-conceptiona)
  - Place of birth
  - Heart Rate (rates/min
  - Respiratory Rate (breaths/min)
  - Follow-up
  - Inherited from father
  - Maternal gene
  - Status
  - Paternal gene
From what values of features are? (Quantifier) then I have chosen values.
For example: with <code>Follow-up</code> feature:  High --> 2, Low ---> 1
- Features have more than 2 values: 
  - It's the same before. But some new values such as: -, Not available, Not applicable,.. so I had to label them.
  - With some text features like : <code>Location of Institute, Institute Name, Family Name, Father's name</code> I have extracted to had new features then encodes them:
    - Location of Institute: 
      for examples: 125 PARKER HILL AV\nJAMAICA PLAIN, MA 02120\n(42.329611374844326, -71.10616871232227). I had created some features before:
      1. 1. JAMAICA PLAIN : district
      2. MA 02120 : POST CODE
      3. 42.329611374844326 : Latitude
      4. -71.10616871232227 : Longtitude
    - Then hash code: <code>district, POST CODE, Family Name, Father's name</code>
## Transforming
- Log transform some numerical features: <code>'Patient Age', 'Blood cell count (mcL)', "Mother's age", "Father's age",
            'White Blood cell count (thousand per microliter)'</code>
- Interaction (ratio): create ratio columns like as in 
    <code>
      df['patient_per_mom'] = df['Patient Age']/df["Mother's age"]
      df['patient_per_dad'] = df['Patient Age']/df["Father's age"]
      df['age_per_bcc'] = df['Patient Age']/df['Blood cell count (mcL)']
      df['age_per_wbcc'] = df['Patient Age']/df['White Blood cell count (thousand per microliter)']
      df['wbcc_per_bcc'] = df['White Blood cell count (thousand per microliter)'] /df['Blood cell count (mcL)'] 
    </code>
- Coordinate features:
    <code>
      lat = df["latitude"]
      lon = df["longtitude"]
      df["x_dimen"] = np.cos(lat) * np.cos(lon)
      df["y_dimen"] = np.cos(lat) * np.sin(lon)
      df["z_dimen"] = np.sin(lat) 
    </code>
## Create IS_NULL some impact features
## Target Encoding
## USING SMOTE TO DEAL WITH IMBALANCE DATASET
# MODEL
Because time was limit, so I had choose [autoML] (https://github.com/mljar/mljar-supervised) for model.
